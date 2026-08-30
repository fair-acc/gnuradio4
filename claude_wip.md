# GPU / heterogeneous-compute integration for GNU Radio 4.0 — design & tracking (V8)

> **Constraint — this file is intentionally NOT git-tracked.** Scratch working doc only: no
> `.gitignore` entry (by request), do not `git add` it, and it will be deleted at the end of this
> branch/PR. Any rationale meant to persist lives in the code (e.g. the `gr::complex` @brief) and
> commit messages, never here.

## 🚧 R0 · BLOCKING REQUIREMENT — device-private block state (maintainer, 2026-09-02)

> **This is a requirement, not a follow-up. It is a design gap in the branch. Fix it, test it and close it
> BEFORE any further spike, filter migration or precondition work.**

> ## ▶ EXECUTION ENTRY POINT — read this order, nothing else (design CLOSED 2026-09-03, no code written yet)
>
> 1. **The decisions D1-D20** below — settled, do not re-litigate.
> 2. **R0.8 — the execution plan.** Four steps. Start at step 1.
> 3. **R0.9 — the tests**, each with the mutant it must kill. Write these RED first.
> 4. **R0.12 / R0.13 / R0.14** — three independent review passes, in that order. R0.14 is the most recent and
>    corrects the two before it.
>
> **SKIP R0.1 – R0.7 entirely.** They describe the withdrawn nested-type design and are kept only as the record
> of why it was rejected.
>
> **State of the branch when this was written:** 32 commits on `syclExperiments`, clean tree, green
> (106/106 `build-acpp`, 102/102 `build-gcc15-debug`), HEAD `b37c68c8`. Nothing in R0 is implemented.
>
> **The design in one paragraph:** kernel-side mutable state is an ordinary **unreflected trivially-copyable
> member** of a span-tier block. It already lives inside `sizeof(TBlock)` and is already carried into the device
> mirror. Three things make it persist: **D3** deletes `copyBackUserState` and the per-dispatch re-seat (which is
> what loses the mutation today), **E1** makes the settings-epoch refresh copy only the `Block` base plus
> reflected members, and **E2** narrows the span-tier canary to reflected members while leaving the full-byte
> compare on the other tiers — which is what keeps a self-mutating block off them. Everything else in R0 is the
> fail-hard/resolver work that must land first, because the state feature is unsafe while any per-call CPU
> fallback survives.

### The requirement, in the maintainer's terms

1. **Blocks sharing one device domain must forward tags and messages on the device.** Forwarding is internal
   `Block<T>` dispatch and data movement, not the processing function's business.
2. **A device `processOne`/`processBulk` must NOT modify settings** — and must not be _able_ to.
3. **A device processing function DOES need to mutate working state**: `last value`, a `HistoryBuffer`, an
   accumulator, a delay line.
4. **That working state need not be host-accessible while the block runs on a device.** It must only be
   **retained between successive calls of the same block's processing function.**

### Cross-check against what the branch actually does (verified 2026-09-02)

**Requirements 2 and 3 are already met.** `ExecutionStrategy.hpp` has **zero** references to `settings()`;
§75 records the deliberate prohibition. Tag- and message-borne settings changes are applied **host-side before
dispatch** -- `settings().autoUpdate()` at `Block.hpp:1352`, then `applyStagedParameters()` -> `++_settingsEpoch`
-> `migrateFieldsToDeviceResource()` at `:1373-1379`, inside `applyInputTagsFromPorts` (called `:2123`), and
`dispatchProcessing` only at `:2295`. Messages arrive via the scheduler (`Scheduler.hpp:537-549`). The kernel
reads the already-updated values through the epoch-triggered mirror refresh. **That is exactly the intended
model and needs no change.**

**Requirement 1 is NOT met.** `forwardInputTags` runs host-side at `Block.hpp:2237`, before dispatch at `:2295`.
Kernels can _publish_ tags (pre-reserved slots), but input->output forwarding is host logic, so a chain sharing
one device domain forwards on the host between dispatches. NOT yet traced: whether this forces a host round trip
for tags on an interior device-only edge -- **establish that before designing the fix.**

**Requirement 4 is NOT met, and this is the core gap.** Every state the branch can persist across dispatches is
host-visible:

| state kind                             | persists across dispatches?                                                                     | what it costs                                                                 |
| -------------------------------------- | ----------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| reflected trivially-copyable           | yes, by round trip through `copyBackUserState`                                                  | becomes a user-writable **settings-surface** member (GRC/YAML, resetDefaults) |
| reflected pmr container of TC elements | contents yes (USM); in-object bookkeeping is host-owned                                         | §80.9 split-state hazard                                                      |
| `HistoryBuffer` (either form)          | **cannot be reflected at all** -- `Block.hpp:2422` fatals "unsupported setting type" (measured) | unusable as device state                                                      |
| unreflected member                     | bit-copied, host pointer followed on device                                                     | broken; the §80.2 gate gap                                                    |

**There is no fourth category: device-private state that persists between dispatches and is not host state.**
`DeviceBlockShadow` already owns a persistent mirror buffer (`DeviceBlockShadow.hpp:18`), so the storage exists;
what is missing is a way to mark state device-private so the mirror does not re-seat it from the host and
`copyBackUserState` does not have to bring it back.

### Why this blocks the rest

- **Precondition (5)** (migrate the filter blocks) is unbuildable as posed: `Section`/`HistoryBuffer` are exactly
  the shape with no slot. The "ring state becomes a user-writable setting" objection is not cosmetic -- it is
  _forced_ by this gap, so no annotation fixes it.
- **Spike A2** (device IIR) is the same shape.
- The §80.9 split-state hazard and the §80.2 gate gap are both symptoms of this one missing category, not
  independent defects.

### ✅ R0 · DECISIONS TAKEN (maintainer, 2026-09-02) — these are settled, do not re-litigate

**D1 — forwarding scope: SPLIT BY TOPOLOGY.**

- **device -> device chain (the 80% case): host-blind.** Tags and messages are pushed via the tag/stream buffer
  **into** the device domain alongside the samples, handled **natively by the blocks in the device domain**, and
  returned to the host at the **end of the chain by a boundary block**. This is what the device-capable
  `ValueMapView` work was for: both tags and messages are ValueMaps, so one transport carries both.
- **single block dispatched to a device (the 20% case): unchanged**, host -> device -> host.

**D2 — SUPERSEDED 2026-09-02: FAIL HARD, no CPU fallback.** _"If a block is supposed to be run on a device but
cannot for whatever reason (e.g. device doesn't exist, settings signature/types incomplete, etc.) be instantiated
then fail hard."_ So there is no migrate-on-fallback, no transfer primitive, no discontinuity signal -- the whole
branch collapses. A block declared for a device domain either runs there or the graph does not start.

**This inverts existing behaviour and is not confined to R0.** Today there are SIX reachable `dispatchCpuFallback`
sites (unresolved domain `ExecutionStrategy.hpp:175`, withdrawn domain `:178`, three mirror-allocation failures
`:433`/`:607`/`:707`, tag-publishing bodies `:501`), all of which warn once and run on the CPU, and tests assert
that behaviour (`qa_DeviceSpans.cpp:501` expects exactly one fallback for an owning-payload block;
`gr::test::cpuFallbacksDuring` exists solely to count them). Converting these to hard failures is its own commit
with its own test migration, and it should move the decision from **per-call dispatch** to **instantiation**:
resolve the domain and prove dispatchability once at init, then never re-decide.

**D2a — ANSWERED 2026-09-02: fail hard AT INSTANTIATION.** Dispatchability is proven once, when the block is
instantiated for its domain; it is not re-decided per call. This fits D-blocks-stay-in-their-domain and removes
the per-call fallback decision entirely.

**Consequence to design against:** several current refusal reasons are only _discovered_ at dispatch today --
whether a body publishes an owning tag payload is found by a probe on first use (`ExecutionStrategy.hpp:501`),
and whether spans are device-resident depends on the edge. Moving to instantiation means each of these must
either become decidable at `init()` or be dropped as a refusal reason.

**D8 — MECHANISM CHOSEN 2026-09-03: the refusal surfaces as the first `work()` returning `ERROR`.** Dispatch
`fail()` already returns `ERROR` and a block returning `ERROR` already escalates the scheduler to `ERROR`
(`Scheduler.hpp:938-940`), so fail-hard costs no new veto path. A true "never reaches RUNNING" would be a NEW
framework mechanism: `Block<T>` defines no lifecycle methods, and `invokeLifecycleMethod` **discards the return
value** (`LifeCycle.hpp:170-190`), so a lifecycle method cannot veto via `std::expected` -- only by throwing,
which is banned. Accepted cost: the graph reaches `RUNNING` for one traversal before stopping.
**REFINED 2026-09-03 -- the free mechanism, which serves the maintainer's choice better than its literal
reading and makes the measurement obligation moot.** Two verified facts change the shape: `checkLifecycle` tests
only `REQUESTED_STOP` and `STOPPED` (`Block.hpp:2065`, `:2073`), **not `ERROR`**, so an ERROR-state block keeps
working; and `dispatchProcessing` runs only after `workInternal` has data (`:2218-2223`), so **a refused block
whose upstream produces nothing never dispatches and the graph completes cleanly** -- the refusal would never be
surfaced at all. Fix both at once: resolve on the `INITIALISED -> RUNNING` transition, set `ERROR` on failure,
and have `checkLifecycle` honour `ERROR`. This still surfaces through `work()` (no new veto path, as chosen), but
it decides and names the block at _start_, closes the never-dispatched hole, and costs a compare on a value
`checkLifecycle` already loads -- so the per-`work()` measurement the maintainer asked for is moot. Note a null
`_deviceContext` cannot serve as the discriminator: a block downgraded to `host` also has a null context with
`_computeDomainIsDevice == true`, since D13 never rewrites the setting.

**Fail hard at SCHEDULER START, not at `emplaceBlock`.** Late wiring stays a legal affordance; the
dispatchability proof runs once when the graph starts. This dissolves the review's biggest risk (nothing in
production calls `registerSyclRuntime()`, so failing at `emplaceBlock` would kill every real graph) and it is
strictly _better_ for the check itself: **edges are connected by then** (`connectPendingEdges` at
`Scheduler.hpp:768-771`), so span types and edge residency are available -- more refusal reasons become
decidable than at `init()`, not fewer.

_Work this implies:_ the scheduler has **no per-block veto of start** today -- it emits an error per block and
continues (`Scheduler.hpp:784-798`). A gate is needed: any block that fails its proof leaves the graph unable to
reach RUNNING.

**D9 — a post-init `compute_domain` change is REFUSED with an error.** Consistent with D8 and with blocks
staying in their domain. _Work this implies:_ a settings-level veto that does not exist yet -- today
`applyChangedSettings` simply re-resolves (`Block.hpp:1374-1377`).

**D10 — PARTLY MOOT 2026-09-03: there is no accessor and no host arm to add** (D16's collapse), so only the
residency half survives. Original text: `DeviceState` residency downgrades per backend, and `deviceState()` gains a host arm.** Device-only
where real device memory exists; host/shared where it does not (`DeviceContextCpu` refuses `devicePtr`,
`DeviceContext.hpp:119-121`); and the same accessor works on a host-domain run backed by ordinary memory. This
keeps the binary portable and, importantly, makes the **host-oracle equivalence invariant testable** -- it could
not be written otherwise, since a host-domain run never allocates a device context. Accepted cost: D5's
"faults loudly" holds only on backends with true device memory; say so where D5 is stated.

**D11 — the ONE permitted fallback: an unavailable domain downgrades to a functional equivalent, PER
SUB-GRAPH, named in the warning.** If a sub-graph declares a device domain the machine cannot serve, only that
incompatible part is downgraded, and the warning names both ends:

    'gpu:cuda' not available, functional fallback to 'host:sycl'

(The maintainer wrote `cpu:sycl`; **that string does not parse** -- `ComputeDomain::mapKind` knows only
`gpu`/`fpga`/`tpu`/`host`, and an unrecognised kind falls through to plain `host()`, i.e. _no SYCL_. The message
must name a domain the reader can paste straight back into `compute_domain` to pin the fallback, so it says
`host:sycl`. Adding `cpu` as a second spelling of `host` was considered and rejected: two names for one thing.)

Other domains in the same graph are unaffected. The fallback target is a _functional equivalent_ -- another
served backend of the same kind where one exists, otherwise the CPU/SYCL one -- not necessarily the plain host.

**Why per sub-graph rather than whole graph:** the sub-graph is the unit in which a compute domain is declared,
so it is the natural unit in which an unmet declaration is resolved. A graph may legitimately end up with one
group on a GPU and another on CPU; that is _declared per group and announced per group_, not the silent
per-block substitution D2 forbids. (An earlier draft of this note proposed all-or-nothing per graph; the
maintainer overrode it, and the reasoning above is why that is right.)

**Why this is compatible with fail-hard, stated precisely:** it is a _graph-scope policy decision taken once at
scheduler start_, not a _per-block silent substitution_. The property the branch actually defends is that no
individual block quietly runs somewhere other than where its neighbours think it does, producing a mixed graph
nobody declared. A uniform downgrade keeps the graph homogeneous and announced. Per-block, per-call fallback
stays forbidden (D2).

**The hole this does NOT close, and it must be stated:** a block whose only device entry point is
`processBulk_sycl` has **no CPU path at all** -- the hatch signature takes a `SyclQueue`. Downgrading its domain
to CPU leaves it undispatchable, so for such blocks the downgrade cannot rescue the graph and D2's hard failure
still applies. The shipped `FFT` carries both a `processBulk` and a `processBulk_sycl` precisely so it survives
this; a hatch-only block would not.

**D12 — registration stays EXPLICIT; there is no auto-registration hook.** An application that wants GPUs
calls `registerSyclRuntime()` itself. An application that only uses the ordinary host dispatch registers
nothing, exactly as before, and must see no new warning. An empty registry is therefore _not_ distinguished from
absent hardware: both are "the machine cannot serve this domain", and both take the D11 downgrade with the named
warning. The warning fires **only when some part of the graph actually declares a device domain** -- a pure host
graph stays silent. Accepted cost: a forgotten `registerSyclRuntime()` runs on the CPU rather than refusing; the
warning is the control, and it names the domain that was asked for.

**D13 — the downgrade is a RESOLUTION-LAYER MAPPING; `compute_domain` is never rewritten.** The setting keeps
saying what the user declared; the resolver maps it to the fallback target for the duration of the run. Three
reasons, each independently sufficient: (a) `compute_domain` **is the IO thread-pool name** (`Block.hpp:737`,
`Doc<"compute domain/IO thread pool name">`, defaulting to `kDefaultIoPoolId`) -- rewriting it to announce a
device downgrade would silently re-point thread-pool selection; (b) it collides with D9, which refuses post-init
`compute_domain` changes; (c) `compute_domain` round-trips through saved graphs, so a rewrite would **persist**
the downgrade -- a graph saved on a laptop would keep running on the CPU after being moved to the GPU machine,
losing the declared intent permanently. Leaner too: a lookup in the resolver, with no setting write, no epoch
bump and no `settingsChanged` storm at start.

**D14 — a device INDEX that does not exist downgrades with the same named warning. CORRECTED 2026-09-03: key
it on the PARSED CANONICAL domain, never on the raw string.** `deviceIndex{-1}` means "provider default"
(`ComputeDomain.hpp:23`) and bare `"gpu"` parses to `{gpu, sycl, -1}` -- a _device_ domain whose raw string is
not a registry key -- so a raw-string rule would warn "'gpu' not available" **on a machine that has a GPU**.
Resolve the canonical string (`kind:backend[:index]`, index omitted when -1) and warn only when a parsed index

> = 0 is unregistered. The message names the canonical domain plus `ctx.name()`, because the canonical `gpu:sycl`
> is the _default queue's_ device, not necessarily enumeration index 0 (`SyclRuntime.hpp:402-412`). Two riders:
> `defaultSyclUsmProvider` performs the same silent index-stripping for edge memory
> (`SyclRuntime.hpp:344-370`) and must resolve identically, or execution and edge memory land on different
> devices; and `Graph.hpp:749` must compare **resolved** strings, since comparing verbatim today already puts a
> host seam between `gpu:sycl` and `gpu:sycl:0` -- two blocks on the same device -- and refuses them as two
> domains.

`gpu:sycl:3` on a
one-device machine today runs on device 0 in silence, because `longestRegisteredPrefixOf`
(`DeviceContextRegistry.hpp:41-52`) strips `:` components until something resolves. That is a within-device-tier
silent substitution and it is closed by treating a missing index exactly like a missing domain:

    'gpu:sycl:3' not available, functional fallback to 'gpu:sycl:0'

**D15 — WITHDRAWN 2026-09-03.** It refused nothing. Its premise was also wrong: `DeviceContext::upload`/
`download` are pure virtual on every backend and used by staging every dispatch (`DeviceContext.hpp:99-100`);
D2 deleted a _policy_, not the transfer primitives. With D4 revised, a restart always passes through `reset()`
and the state is re-seeded there, so there is nothing for a start-time comparison to protect. Deleted rather
than kept as a no-op.

**D16 — REPLACED 2026-09-03 by THE COLLAPSE: no nested type, no accessor, no new surface.** The review
established that `KernelState` is not distinguishable from an unreflected `mutable` member. An unreflected
trivially-copyable member written by a span body fails to persist today for exactly two reasons beyond D3:
the epoch refresh memcpys the whole object over the mirror (`ExecutionStrategy.hpp:251`), and the span-tier
canary compares all `sizeof(TBlock)` bytes. So the feature is TWO EDITS inside existing functions:

- **E1** — the epoch refresh copies the `Block<Derived>` base **plus the reflected user members** (the reflection
  loop already exists in `copyBackUserState`), instead of the whole object. Unreflected members are seated once
  on `acquire` and never again, so a settings change no longer clobbers them.
- **E2** — the span-tier canary compares **reflected members only**; the auto-parallel and view canaries keep the
  full-byte compare, which enforces the tier restriction **by construction** rather than by a `static_assert`.

Then `HistoryBuffer<float, 16>` works as an unreflected member (trivially copyable, `HistoryBuffer.hpp:70-77`;
the contracts gate inspects only reflected members, `Block.hpp:2411-2428`), the host-domain oracle works with no
host arm, and D4's re-seed is `shadow.epoch = kNeverRefreshed` + "full copy when never refreshed". ~20 lines of
framework change against the ~150 the nested type implied. Knowingly given up: a
`static_assert(is_trivially_copyable)` on the state, and a greppable type name. Requirement 4 is met by a
documented guarantee plus the R0.9 tests.

**D17 — D11's fallback ladder is FIXED, with no same-kind search** (maintainer, 2026-09-03):

    index-stripped canonical (kind:backend) -> host:sycl -> host

**"Canonical" means the INDEX-STRIPPED `kind:backend`** (maintainer, 2026-09-03), which resolves D17's
contradiction with D14: `gpu:sycl:3` on a one-GPU machine falls back to `gpu:sycl` and stays on the GPU, rather
than to the SYCL CPU. The `host:sycl` rung would have been unsafe there -- two such blocks get a **DeviceOnly**
interior ring on GPU 0 (`Graph.hpp:750-751`) that a CPU kernel would then read. The `host` rung exists for the
no-backend build (`Block.hpp:1996-2000`), where `host:sycl` is unserved too.
No preference order is needed and the result is deterministic; the cost is that a machine serving `gpu:hip`
falls back to the host when `gpu:cuda` was asked for, rather than using the GPU it has. A same-kind scan can be
added later behind a measurement, but it would need a defined order (map order is not deterministic) and
coherent substitution across BOTH registries or placement splits.

**D18 — CONFIRMED 2026-09-03: the discriminator between refusal and downgrade is WHETHER THE DOMAIN
IS SERVED, not why the block cannot run.** Two rules, no third category:

- the **domain is unserved** (no such device, no such index, no backend compiled in) -> **D11 downgrade**, named.
- the **block cannot use a SERVED domain** (its type has no device entry point for this `T`) -> **D2 refusal**.

This is not a new decision so much as the original fail-hard instruction applied literally -- _"if a block is
supposed to be run on a device but cannot for whatever reason (e.g. device doesn't exist, settings
signature/types incomplete, etc.) be instantiated then fail hard"_ already names the types-incomplete case. It
also forces the right answer for the no-backend build: with nothing compiled in the domain is unserved, so that
path downgrades, and a gcc-compiled TU does not refuse every device graph.

VERIFIED costs: `FFT::processBulk` is non-const (`fft.hpp:83`, `:203`), so `FFT<double>` has no device path and
`Block.hpp:1957` is reachable -- but **no test or example pairs `FFT<double>` with a device domain**, so the
refusal breaks nothing today. Residual cost: a pipeline templated on `T` starts at `float` and refuses at
`double`, and the author's fix is to declare `host` for the double instantiation. The refusal message must name
that fix.

Maintainer's rationale on accepting the residual cost: double precision on a device is a rare use case anyway --
GPUs are normally single-precision -- so a pipeline that refuses at `T = double` is refusing a configuration
that would have been a poor choice regardless.

**D19 — domain identity: NORMALISE THE CANONICAL NAME AT REGISTRATION** (maintainer, 2026-09-03). Register one
context per device under one canonical name; other spellings resolve to it. Today `registerContext` creates a
**separate `DeviceContextSycl` per name** (`SyclRuntime.hpp:158`), so `gpu:sycl` and `gpu:sycl:0` are two objects
on one queue, and `Graph.hpp:750`'s verbatim string compare therefore puts a **pinned host ring between two
blocks on the same GPU** and makes `refuseTwoDeviceDomains` count two domains. Normalising at registration keeps
string comparison working everywhere it is used today and is the smaller change to `Graph.hpp`. Accepted cost:
the registry must know which names are aliases, and the fix does not generalise to non-SYCL backends -- if a
second backend ever registers this way, revisit and compare the queue instead.

**D20 — two small ones taken by default, stated so they are not re-derived:**

- a **`reset_default` tag does NOT clear kernel state.** The re-seed is keyed on the lifecycle _transition_, not
  on the `reset()` method, because `reset()` is also invoked mid-RUNNING by a `reset_default` tag/message
  (`Settings.hpp:1043-1045`, `:1204-1208`) -- keying on the method would silently zero a running filter's delay
  line. A block that wants the tag to clear its state does so in its own `settingsChanged`.
- **"served" is per TU as well as per process:** a domain is served iff the registry resolves it **and this
  translation unit compiled a backend**. In a gcc-only TU of a mixed binary the registry may report `host:sycl`
  served while that TU cannot dispatch at all (`Block.hpp:1996-2000`), so the D17 ladder collapses straight to
  `host` there and the warning names the missing backend rather than the missing device.

**Hook point (settled by elimination, not preference): `Block<T>::changeStateTo`.** Both D4's re-seed and D8's
resolution hang there, ~6 lines forwarding to the state machine. `reset()` cannot be used -- 25 Derived types
define one and CRTP hides the base -- nor `stateChanged()` (hidden by `BlockMerging.hpp`). Every path goes
through `changeStateTo` (`BlockModel.hpp:843`, `Block.hpp:973`, `:1489`, `SubGraph.hpp:62-63`) and no block in
the tree defines its own.

**D3 — legacy path: REMOVE once the filters migrate.** Kernel-written _reflected trivially-copyable_ state with
`copyBackUserState` goes away; an **unreflected trivially-copyable member** becomes the one way to keep state. (Stronger than the analysis
proposed — it retires the §80.9 and §80.18 hazard classes outright instead of leaving them audible.)

**D7 — a processing function must not change settings, and must not be able to.** _"That's what the
settingsChanged(..) methods and tags are for."_ So after D3 the span tier must REFUSE a body that mutates a
reflected member, matching what the auto-parallel (`ExecutionStrategy.hpp:698`) and view (`:642`) tiers already
do. Without this, post-D3 such writes would go to the mirror and be silently discarded -- exactly the class of
silent loss §80.18 exists to prevent. An **unreflected member** is then the ONLY legal mutation target for a kernel body.

**D4 — lifecycle: REVISED 2026-09-03. Preserved across pause/resume and shape-preserving settings changes;
re-seeded on entry to `INITIALISED`.** The original "preserved across `start()`/`stop()`" row was not
implementable: `isValidTransition` permits `STOPPED -> INITIALISED` only (`LifeCycle.hpp:108`), and any
`-> INITIALISED` from a non-IDLE state invokes `reset()` (`:266-268`), so a block cannot reach `RUNNING` a second
time without passing through `reset()` -- by the state machine alone, independent of the scheduler. Preserve and
zero-on-reset can therefore both hold only for pause/resume, which calls neither. This matches the documented
contract at `Block.hpp:625`.

**D5 — SUPERSEDED 2026-09-03 by D16's collapse: there is NO separate state buffer, and nothing to over-allocate.**
Kernel-side state is an ordinary **unreflected, trivially-copyable member of the block**. It already lives inside
`sizeof(TBlock)` and is therefore already carried into the mirror and seated once on `acquire`. The earlier
"ride the mirror by over-allocating a trailing region" scheme was needed only for a _separately declared_ nested
type; with the collapse it has no purpose. Device-only residency remains a measured follow-up and is still one
enum (`DeviceBlockShadow.hpp:44`).

**D6 — device-side tag forwarding: BUILD IT AS PART OF R0**, not deferred behind the counters. **Reconfirmed
2026-09-03** against the finding that the host hop is per tag _event_, not per sample (samples never touch the
host on an interior device-only edge; tag bytes are host-touched three times per hop, `Block.hpp:1240-1243` and
`ExecutionStrategy.hpp:507-534`). Definition-of-done item 2 stays a gate on R0.

**Tier restriction: CONFIRMED.** A state-keeping block is span-tier only. Under the collapse this is enforced
**by construction**: E2 leaves the full-byte canary on the auto-parallel and view tiers, so a block that mutates
itself cannot reach them.

**Tag-ring residency: default SYCL USM via the USM-PMR resource** (host- and device-accessible), which makes the
boundary case easy. A **device-only** PMR for interior device->device edges is an _optimisation_ that must be
demonstrated by measurement first -- not part of the default design.

#### R0.0 What this changes about the work — verified 2026-09-02

**The tag memory is ALREADY where D1 needs it.** `ComputeDomain.hpp:16` defines
`Access::{HostOnly, Shared, DeviceOnly}`, and `Graph.hpp:767` resolves a device edge's tag axis with
`tagDomain.access = Access::Shared` -- i.e. shared USM, host- **and** device-accessible. So requirement 1 needs
**no memory-placement change by default**; the ring is already reachable from a kernel.

What is actually missing for D1 is the _handling_, not the storage:

1. block-side tag read/write from inside a kernel over the shared-USM ring (the pieces exist: kernels already
   publish through pre-reserved slots, and `ValueMapView` is device-callable);
2. moving `forwardInputTags`' pass-through logic device-side for interior edges (`Block.hpp:2237` runs host-side
   today, before dispatch at `:2295`);
3. **the boundary block** that drains tags and messages back to the host at the end of a device chain -- this is
   new and is the piece D1 names explicitly;
4. messages reaching the device domain at all (today they are scheduler-routed host-side only,
   `Scheduler.hpp:537-549`, with no per-edge substrate).

**The rationale comment at `Graph.hpp:766` is now stale** -- "a kernel may not publish tags" was true when
written; kernels publish tags today through pre-reserved slots. The _outcome_ (shared USM) is right for D1; only
the stated reason is obsolete. Fix the comment when this work lands.

### 📘 R0 · THE DESIGN (analysed 2026-09-02; adversarial review + code verification)

> ## ⛔ R0.1 – R0.7 BELOW DESCRIBE THE **WITHDRAWN** NESTED-TYPE DESIGN (superseded 2026-09-03)
>
> They are kept ONLY as the record of what was considered and why it was rejected. **Do not implement from
> them.** There is no nested type, no `DeviceState`/`KernelState`, no accessor, no separate allocation, no
> `DeviceProbeSafe` exclusion and no PoC milestone. Every mention of `deviceState()`,
> `requestDeviceStateReset()`, `DeviceStateIsReflected` and "one persistent device allocation for it" is dead.
>
> **The live design is: decision D16 (the two edits E1/E2) + D3 + D5-superseded, and the plan is R0.8.**
> Kernel-side state is an ordinary **unreflected trivially-copyable member of the block**, already inside
> `sizeof(TBlock)` and already carried by the mirror.
>
> The one thing in R0.1-R0.7 that survives and still matters: the **tier restriction** (state-keeping blocks are
> span-tier only, because N work items sharing one mutable state is a race). Under the collapse this is enforced
> **by construction** -- E2 leaves the full-byte canary in place on the auto-parallel and view tiers, so a block
> that mutates itself cannot reach them -- rather than by a `static_assert` on a declared type.

#### R0.1 Recommended shape — a declared nested `DeviceState` type ⛔ SUPERSEDED

The block declares a nested type; it does **not** declare a member of that type:

```cpp
struct MyFilter : gr::Block<MyFilter> {
    gr::PortIn<float> in; gr::PortOut<float> out;
    Annotated<float, "cutoff"> cutoff = 1.f;      // ordinary setting, host-owned
    GR_MAKE_REFLECTABLE(MyFilter, in, out, cutoff);

    struct DeviceState {                          // device-private working state
        gr::HistoryBuffer<float, 16> delay{};     // template-static form: trivially copyable
        gr::Size_t                   calls = 0U;
    };

    gr::work::Status processBulk(InputSpanLike auto& i, OutputSpanLike auto& o) {
        auto& s = this->deviceState();            // same accessor on host and device
        ...
    }
};
```

`Block<T>` detects `typename Derived::DeviceState` by concept, owns ONE persistent device allocation for it
beside `DeviceBlockShadow`, seeds it once from a default-constructed host image, and never re-seats or copies it
back.

**Why this and not the alternatives** — the decisive property is that all four exclusions hold _by
construction_, not by a predicate someone must keep in sync:

|                                  | settings surface     | `checkBlockContracts`                 | `copyBackUserState`    | re-seat on epoch                       |
| -------------------------------- | -------------------- | ------------------------------------- | ---------------------- | -------------------------------------- |
| **nested `DeviceState`**         | nothing to enumerate | never sees it (not a member)          | nothing to do          | outside `sizeof(TBlock)`               |
| member marker (`DeviceLocal<T>`) | by predicate         | needs a carve-out at `Block.hpp:2422` | needs a carve-out      | **needs segmented relocate**           |
| pmr / allocator                  | n/a                  | n/a                                   | skips it (§80.9 split) | contents survive, bookkeeping does not |
| mirror-authoritative             | n/a                  | n/a                                   | removed                | forks the model per tier               |

- **member marker loses** because avoiding the re-seat means replacing the single `memcpy`
  (`DeviceRelocatable.hpp:136-138`) with a member-wise walk, which cannot copy _unreflected_ members at all.
- **pmr loses** because it IS §80.9 restated, and `rebindFieldsTo` seats every pmr field onto ONE resource
  (`Block.hpp:1496-1511`), so mixed residency (coefficients shared, state device-only) has no mechanism.
- **mirror-authoritative loses** on blast radius and because it reverses the `mirrorStateReturns` fix.

`HistoryBuffer` inside `DeviceState` works because the contracts gate never sees it -- which closes R0's
definition-of-done item 4 for free.

#### R0.2 How the kernel reaches it ⛔ SUPERSEDED

One raw pointer in the `Block<T>` base (unconditional size, see R0.6) + the persistent allocation beside the
shadow. Before each relocate the dispatcher writes the current device pointer into the host object; the existing
`memcpy` carries the **value**; the body reads it back through `deviceState()`.

- USM pointers are plain addresses, valid in-kernel; the known trap is context-boundness (cross-context use
  segfaults), handled under migration in R0.3.
- `static_assert(is_trivially_copyable_v<DeviceState> && is_default_constructible_v<DeviceState>)` at the
  detection site -- admits the template-static `HistoryBuffer`, correctly excludes the pmr form.
- **Alignment trap:** `DeviceContextCpu` refuses `align > max_align_t` (`DeviceContext.hpp:121-123`), so an
  `alignas(kCacheLine)` state struct would fail to allocate on the CPU context. Either lift that or document the
  cap; do not let it fail silently.
- **`DeviceState` is span-tier only.** N work items sharing one mutable state is a race, so the auto-parallel and
  view arms must refuse such a block at compile time. This _aligns_ with the measured rule "stateful => span
  tier" rather than fighting it.
- **`DeviceProbeSafe` must exclude `DeviceState` blocks**: the mutation canary and tag-publish probe run the real
  body on a host-side bit-copy, and a body dereferencing a device-only pointer must never run there.

#### R0.3 Lifecycle — device state behaves like an ordinary member, and is loud where it cannot ⛔ SUPERSEDED

| event                                      | behaviour                                                                                                                              |
| ------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------- |
| first use                                  | lazy allocate + seed from a default-constructed image                                                                                  |
| `start()`                                  | **preserve** (host semantics)                                                                                                          |
| `stop()`                                   | preserve bytes, do not free -- a stop/start cycle must not glitch                                                                      |
| `reset()`                                  | **zero** (lifecycle contract)                                                                                                          |
| settings change, shape-preserving (cutoff) | **preserve** -- the whole point                                                                                                        |
| settings change, shape-changing (order)    | preserve by default; the block's `settingsChanged` calls `requestDeviceStateReset()`                                                   |
| block move                                 | state lost; moved-to block re-seeds lazily (moves happen at graph assembly)                                                            |
| teardown                                   | released with the shadow in `~Block`                                                                                                   |
| domain migration                           | download old -> host staging -> upload new while the old context is `served()`; otherwise re-seed + the discontinuity signal from R0.4 |

#### R0.4 CPU fallback — the tension, and the recommendation ⛔ SUPERSEDED

Fallback is **per-call and interleaving is live**: six reachable sites (unresolved domain, withdrawn domain,
three mirror-allocation failures, tag-publishing bodies) and nothing latches it. Today this is harmless because
the host object is authoritative every dispatch. Device-private state breaks that symmetry.

**Recommended: migrate on fallback, terminal error when unrecoverable.** Download state on device->host
transition, upload on return: bit-exact continuity, cost only at transitions, and mechanically the SAME transfer
primitive domain migration needs -- one mechanism, two callers. When the bytes are gone (device lost), raise
`gr::Error` rather than silently re-seeding; a state-bearing DSP block that quietly restarts its recursion is
worse than a stopped graph. An opt-in per-block policy may choose re-seed + a discontinuity tag instead.

Rejected: _refuse once live_ punishes recoverable hiccups and forbids the structural fallback that lets one
binary run anywhere; _accept the glitch_ institutionalises the silent wrongness this branch exists to refuse.

#### R0.5 Three mechanisms, not one — §75's "build it once" assessed ⛔ SUPERSEDED

- **device tag publish** and **device->host reporting** ARE one pattern (slab + post-barrier drain); unify them.
- **device-private state is NOT a transport**: retained, never drained, no host reader. Forcing it into the slab
  would give it drain semantics it must not have.
- **device-side forwarding** is a third thing: edge-scoped, host-classified, device-copied. Gate it on the §80.0
  transfer counters -- tags are per-event where samples are per-element, so the saving may be noise.

**Hard prior:** `DeviceLog.hpp:44-46` -- host and device must never write a slab concurrently (cross-device
atomics are not coherent), so "the kernel publishes into the downstream ring" is off the table. Any device-side
forwarding is host-orchestrated, device-executed.

**The tension to resolve (D1):** requirement 1 collides with the requirement-2 model already blessed. Tag-borne
settings are applied host-side pre-dispatch, which requires the host to read every input tag of every block every
work call. Fully host-blind forwarding is therefore impossible without reopening the settings model. The
reconcilable form is **control stays host, payload stays device**.

#### R0.6 Blast radius ⛔ SUPERSEDED

Touched: `Block.hpp` (base pointer + accessor + lifecycle hooks), `DeviceBlockShadow.hpp` (second buffer + reset
flag), `ExecutionStrategy.hpp` (seat/seed, tier refusals, fallback migration), `DeviceRelocatable.hpp` (one
`DeviceProbeSafe` conjunct), `USER_API_GPU_Blocks.md`. **No change** to `checkBlockContracts`, `Settings.hpp`,
reflection or `Graph.hpp`.

**One-layout/ODR constraint is binding** (`Block.hpp:807-810`): the new members must be unconditional-size like
`_deviceShadow` (~48 B), because `GR_DEVICE_HAS_ANY_BACKEND` differs between acpp- and gcc-compiled TUs linked
into one binary. Do not gate them behind the backend macro.

Production blocks changed: **none** (no shipped block keeps span-tier mutable state). Test blocks
`ZeroCrossingTrigger`, `ZeroCrossingTriggerView`, `RunningTotal` migrate -- keep one on the legacy path
deliberately, so the §80.18 stale-mirror test keeps testing it.

#### R0.7 Proof of concept — before any framework edit ⛔ SUPERSEDED

One test TU on `gpu:sycl`, no core changes: a block with `struct DeviceState { HistoryBuffer<float,16> delay;
Size_t calls; }`, hand-rolled seat/seed, pointer smuggled through an unreflected member (legal today via the
§80.2 gap). Assert: output equals a host-oracle run (state persisted with zero copy-back); a tag-borne settings
change mid-run leaves the sequence unbroken while the setting takes effect; reflecting the pointer is refused by
the relocatable gate.

**Falsifiers — any one reshapes the design:** (F1) the relocated mirror's pointer does not dereference in-kernel
-> state must become a kernel argument and the concepts grow a parameter; (F2) per-dispatch upload races
`parallelFor` on the in-order queue -> needs an explicit event chain; (F3) device-only allocation of a few
hundred bytes fails or is pathologically slow -> flip default residency to shared; (F4) the settings path frees
or moves something the pointer depends on.

#### R0.8 EXECUTION PLAN — settled 2026-09-03, supersedes every earlier sequencing note

**The PoC milestone is GONE.** It existed to retire the risk "does a smuggled pointer survive the mirror and
dereference in a kernel". With D16's collapse there is no pointer: the state is an ordinary unreflected member
inside `sizeof(TBlock)`, which the mirror demonstrably already carries -- that is how settings reach the device
today. What remains worth proving early is that such a member _retains its value across dispatches_, and that is
R0.9 test 1, written RED first (§80.18's lesson: write it against the mutant before the fix exists).

**Step 1 — ✅ LANDED 2026-09-03** (5 commits, both gates green: acpp 106/106, gcc15-debug 102/102).
`66d8f200` resolver ladder · `9c250143` start-time decision · `0a9ea1ee` refusal + latch ·
`a5d6e01d` two-device group refuses · `4d555e48` FFT host-span fence.

**What the work changed about the plan (corrections earned by building it):**

- **"no backend is wired" is a DOWNGRADE, not a refusal.** Converting it to fatal broke `qa_Block` — an empty
  registry is exactly D12's unregistered case. The latch is what resolves it: `_deviceContext` is set once when
  the block starts and dispatch gates on _that_, so a downgraded block simply runs its host body and never
  attempts dispatch. `Block.hpp`'s dispatch gate is now `_deviceContext != nullptr`, not `_computeDomainIsDevice`.
- **The refusal must resolve to the OWNER name, not a yes/no.** An alias makes two spellings one _context_ but
  not one _string_, and `Graph.hpp:749` compares strings — so a bool predicate would have left `gpu:sycl` and
  `gpu:sycl:0` split. `resolveComputeDomain` takes an owner lookup; `downgraded` keys on which rung answered, so
  normalising within a rung is not reported as a downgrade.
- **Keep the refusal telemetry.** 21 fallback assertions existed, but 18 expect ZERO — they are "the kernel
  really ran" guards. Deleting `dispatchCpuFallback` outright would have broken all 21 for no gain; making it
  fatal while still announcing kept their meaning verbatim. Renamed `cpuFallbacksDuring` ->
  `deviceRefusalsDuring`. **The matched log text is a contract** — this is what silently zeroed the counter in the
  earlier unexplained bisect, and it is now documented at both ends.
- **Actual break count: 3 of 106** (`qa_Block`, `qa_DeviceSpans`, `qa_DeviceBlockStyles`) — `qa_Block` was NOT
  predicted, and its two failures were the honest ones: they encoded the warn-once-and-fall-back contract.
- **A refused block surfaces as a thrown exception** when nothing subscribes to the message port
  (`Scheduler.hpp:562`, pre-existing escalation). Tests use `gr::test::runAbsorbingRefusal` and assert on the
  scheduler's final state instead of on how the error surfaced.
- **`runAndWait()` returns success even when the run ended in ERROR.** Assert on `sched.state()`, as
  `qa_HttpBlock.cpp:138` already does. Cost me one wrong assertion.
- **DEFERRED, needs a decision:** the downgrade warning fires **per block**, not once per graph. A 200-block GPU
  graph on a CPU machine emits 200 identical lines. Per-block _does_ name the offending block, which is useful;
  consolidating needs either a scheduler-level pre-pass or shared per-run state, and the existing per-type
  `static` flag is unsuitable (two graphs in one process share it). Left as-is deliberately — ask the maintainer.

**Step 1 (original plan text below, for reference).**
Lands D2/D2a, D8, D11-D14, D17-D20. This must precede the state work: the state feature is _unsafe_ while any
per-call fallback survives, because a mid-run host fallback would run the host body against a device pointer.

- one resolution function `declared -> {resolvedDomain, DeviceContext*}` beside `ComputeDomain.hpp` (must compile
  without device headers); all THREE consumers call it -- dispatch (`Block.hpp:1968`), pmr seating (`:1006-1007`)
  and edge placement/classification (`Graph.hpp:747-751`, `:769-771`)
- normalise canonical names at registration (D19); resolve the parsed canonical, never the raw string (D14)
- latch: drop `Block.hpp:1376`; `ExecutionStrategy.hpp:175-181` -> `fail()` (the dead withdraw arm goes with it,
  since `DeviceContextRegistry::withdraw()` has no caller); `migrateFieldsToDeviceResource` uses the resolved
  string instead of early-returning on "no backend yet"; a generation counter on both registries recorded at start
- resolve on the `INITIALISED -> RUNNING` transition in `Block<T>::changeStateTo`; `ERROR` on failure;
  `checkLifecycle` honours `ERROR`
- `refuseTwoDeviceDomains` becomes a start refusal (`SubGraph.hpp:57`), matching `makeSubGraph:284`
- warn-once per group at `SubGraph::startDispatch` (`SubGraph.hpp:55`) and in `Scheduler::start` **before**
  `connectPendingEdges` (`Scheduler.hpp:769`) for bare blocks -- NOT the per-type `static` flag at
  `ExecutionStrategy.hpp:62-66`, which two graphs in one process would share
- **FFT fix (maintainer: fold in here):** constrain `FFT::processBulk` to host spans as `TransferBlocks.hpp:27-31`
  already does, so only the float hatch reaches the device. Step 2's const flip makes this structural, but the
  explicit constraint documents intent and does not depend on tier rules -- keep it
- KEEP `bulkPublishesTags`, `firstUse` and the `mutatesItsOwnState` family; step 2 reuses them
- KEEP the pre-launch owning-tag probe: without it the refusal comes from the post-kernel flag (`:543-546`) after
  a kernel has already tried to build a `property_map` on the device -- a fault, not an ERROR
- **KEEP the announcement/counting mechanism; make the fallback FATAL rather than deleting the path's telemetry.**
  Measured 2026-09-03: there are **21 fallback assertions across 4 test files**, but **18 of them expect ZERO**
  -- they are guards meaning "the kernel really ran, it did not quietly fall back"
  (`qa_DeviceAutoParallel.cpp` alone has 8, plus `qa_DeviceSpans.cpp:494,547,566,630`,
  `qa_DeviceBlockStyles.cpp:292,310`, `qa_DeviceSeam.cpp:69-71`). If `dispatchCpuFallback` is deleted outright
  the `gr::test::cpuFallbacksDuring` helper (`device_test_helpers.hpp:49-63`) has nothing to count and all 21
  call sites stop compiling. Instead **rename the concept to a refusal**: the site returns `ERROR` and still
  announces, so the 18 zero-expectations keep their meaning **verbatim** and only the positive ones move.
- tests actually inverting: **3 assertions in 2 files** -- `qa_DeviceSpans.cpp:501` (expects exactly 1) and
  `qa_DeviceBlockStyles.cpp:252` (expects > 0); `qa_DeviceBlockStyles.cpp:257` tests fallback _behaviour_ ("the
  CPU fallback must serve every declared port") and that case is deleted, not inverted.
- new regression: two adjacent `compute_domain="gpu"` blocks (today they run the CPU fallback body over a
  device-only ring, because `tryResolve("gpu")` has no `:` so the prefix loop never runs)
- new regression: `gpu:sycl` beside `gpu:sycl:0` must NOT produce a host seam or a two-domain refusal

**Step 2 — ✅ R0 MVP LANDED 2026-09-03** (`3c046361` + `bf34dc25`, both gates green: acpp 106/106,
gcc15-debug 102/102). **The requirement is met: a block can keep mutable state on the device across dispatches,
unreflected, without copy-back.**

**What actually shipped, and what did NOT:**

- **E1 shipped** as `refreshDeviceSettings` (`DeviceRelocatable.hpp`): the first seat is a whole-object copy, every
  later one copies only the trivially-copyable _reflected_ members. Unreflected members therefore survive a
  settings epoch bump. **D3 shipped**: the per-dispatch `mirrorStateReturns` re-seat and `copyBackUserState` are
  both gone — the re-seat, not the epoch, was what actually lost the mutation.
- **E2 turned out to need NO CODE.** The plan had the span-tier canary narrowed to reflected members. Not needed:
  the auto-parallel and view tiers keep their full-byte compare, and that is precisely what keeps a self-mutating
  block off them — the tier restriction is enforced by construction, as intended, with nothing to write.
- **The const span signature (D7) did NOT ship** and is still open. A span-tier body remains non-const, so a
  kernel can still write a _reflected_ member; post-D3 that write now persists in the mirror until the next
  settings change and is then silently reset — the §80.18 shape. Needs the const flip plus the S3 canary.
- **The D4 lifecycle re-seed did NOT ship.** State currently survives a stop/start because the mirror is only
  released when the context changes. `shadow.epoch = kNeverRefreshed` on entry to INITIALISED, via the
  `Block<T>::changeStateTo` hook that already exists from step 1, is the remaining work.

**Two API facts worth not rediscovering:** `settings().set()` stores against a _timestamp_ context and does NOT
stage — a mid-run change made that way never applies. Use `settings().setStaged()`. And the discriminating test
is arithmetic, not an inequality: N samples at gain 1 then M at gain 4 must total `N + 4M`; a refresh copying the
whole block gives `4M` (state reset), one copying nothing gives `N + M` (setting ignored).

**Step 2 (original plan text below, for reference).**
It cannot be split: the const flip breaks `ZeroCrossingTriggerView` (asserted to run as a kernel,
`qa_DeviceSpans.cpp:630-637`) and `RunningTotal` (`:439`), and only E1/E2 restore them.

- D3: delete `copyBackUserState` and the per-dispatch `mirrorStateReturns` re-seat -- this is what loses the
  mutation today, not the epoch
- const span signature (D7); `Upsampler` (`:278`) and `WeightedDifferenceSpans` (`:317`) need only a `const` added
- **E1**: epoch refresh copies the `Block<Derived>` base + reflected user members, not the whole object
- **E2**: span-tier canary compares reflected members only; auto-parallel and view keep the full-byte compare,
  which enforces the tier restriction by construction
- re-seed: `shadow.epoch = kNeverRefreshed` on entry to `INITIALISED`, "full copy when never refreshed" on
  `acquire` -- seed on next acquire, NOT "zero the current mirror", or the first-start case (`Block.hpp:973`
  enters INITIALISED from `init()` with no mirror yet) is missed
- a state-declaring functor without a persistent shadow (`!kOwnsDeviceShadow`, `ExecutionStrategy.hpp:257`) must
  be a compile error
- R0.9 tests 1-4, 6, 8, each written against its mutant

**Step 3 — ✅ SPIKE A2 LANDED 2026-09-03.** A real IIR biquad section (`DeviceIirSection`, `qa_DeviceSpans.cpp`)
runs as one work item on the device and agrees with the identical host body sample for sample, over many small
dispatches, with zero refusals. **The R0 design is validated end to end by a real DSP block.**

**What it settled, beyond passing:**

- **The coefficient question is answered: `std::pmr::vector` reflected + a fixed-size `mutable` delay line.**
  Coefficients are a reflected setting the host owns, re-seated onto device memory by the existing pmr migration;
  the delay line is `mutable std::array<float, kMaxOrder>`, unreflected, trivially copyable, riding the mirror.
  This is the shape to migrate real filters into.
- **The existing `iir_filter` cannot be migrated as-is**, for two independent reasons: `std::execution::unseq` in
  `std::transform_reduce` is host-only, and `HistoryBuffer<T>` with runtime capacity owns heap storage whose
  pointer would be followed onto the device. A device section wants a compile-time maximum order and cascades
  rather than widening — which is also the answer to the maintainer's earlier question about static section sizes.
- **A limit of E1 worth knowing:** only _reflected_ members are refreshed on a settings epoch. An unreflected
  member that the HOST writes (e.g. coefficients packed into a fixed array by `settingsChanged`) would never
  reach the device. Unreflected means kernel-owned; host-owned data must be reflected, or pmr, or both.

**Step 4 — device-side tag forwarding (D6) + slab unification.** Definition-of-done item 2. The host hop is per
tag _event_, not per sample.

**Follow-ups, each gated on measurement:** device-only residency for the mirror (one enum,
`DeviceBlockShadow.hpp:44`) -- and the measurement MUST include a Debug build, because `staleMirrorDiagnostic`
(`ExecutionStrategy.hpp:435`) reads the mirror every dispatch there, which is exactly the host-reads-what-the-
kernel-wrote pattern measured at 15x on the FFT streaming path. Transfer counters are their own milestone and
are NOT commit 1 -- they existed on this branch and were removed as dead API (d1fa1d7a, 99b0f0ef).

#### R0.9 Tests, each with the mutant it must kill — REWRITTEN 2026-09-03 for the collapse

Every one written RED first, against the mutant, before the fix exists (§80.18 was nearly shipped with a test
that passed without its fix). The state under test is an **unreflected trivially-copyable member**, not a
declared type.

**Step-1 tests (resolver, latch, fail-hard):**

1. **two adjacent `compute_domain="gpu"` blocks** -- today they run the CPU fallback body over a device-only
   ring (`tryResolve("gpu")` has no `:`, so the prefix loop never runs). Mutant: resolve the raw string.
2. **`gpu:sycl` beside `gpu:sycl:0`** -- must be one domain on one device: no host seam, no two-domain refusal.
   Mutant: compare declared strings verbatim.
3. **`gpu:sycl:3` on a one-GPU machine** -- downgrades to `gpu:sycl` with the named warning and STAYS on the
   GPU. Mutant: the `host:sycl` rung (which would give a CPU kernel a device-only ring).
4. **a refused block whose upstream produces nothing** -- the graph must NOT complete cleanly. Mutant: put the
   refusal in `dispatchProcessing`, which never runs without data.
5. **`FFT<double>` on `gpu:sycl`** -- must not reach a kernel. Mutant: today's unconstrained `processBulk`,
   which bit-copies four `std::vector`s to the device.
6. **warn-once is per graph, not per type** -- two graphs in one process each warn. Mutant: the per-type
   `static` flag at `ExecutionStrategy.hpp:62-66`.

**Step-2 tests (the state feature):** 7. **an unreflected member persists across dispatches.** Mutant: today's per-dispatch re-seat
(`mirrorStateReturns=true`, `ExecutionStrategy.hpp:251`) -- which is the real reason it does not persist. 8. **it survives a settings-epoch bump, AND the changed setting still takes effect.** Mutant: E1 copying the
whole object (today's behaviour) -- catches a fix that over-corrects and stops applying settings. 9. **it never appears as a setting** -- assert the settings snapshot equals the declared settings. 10. **`HistoryBuffer<float,16>` as state, oracle = the same body on `compute_domain="host"`** -- the equivalence
invariant. Mutant: drop the state carriage. Run on `host:sycl` and the CPU context too, not only `gpu:sycl`. 11. **tier refusal** -- a self-mutating block cannot reach the auto-parallel or view arm. Mutant: relax the
full-byte canary on those tiers (which is what would silently re-open the N-work-items race). 12. **re-seed on entry to `INITIALISED`, preserved across pause/resume** (D4). Mutant: re-seed on every start
_including the first_, or zero the current mirror instead of marking "seed on next acquire" -- the latter
misses the first-start case, where `Block.hpp:973` enters INITIALISED with no mirror yet. 13. **a `reset_default` tag does NOT clear it** (D20). Mutant: key the re-seed on `reset()` instead of the
lifecycle transition. 14. **teardown: the mirror is released exactly once** (`gr::CountingResource`).

Deferred until the counters exist: zero host payload copies for pass-through tags.

#### R0.10 RE-REVIEW FINDINGS (independent, 2026-09-02) — corrections to R0 above

**LIVE DEFECT, verified, independent of R0:** `DeviceContextSycl::allocate(std::size_t bytes, std::size_t
/*align*/, Residency)` **discards the alignment argument**. `shadow.acquire(ctx, sizeof(TBlock),
alignof(TBlock))` passes 64 (measured: `alignof(Block-derived) == 64`, `alignof(max_align_t) == 16`), so every
device mirror is under-aligned by contract today. It has not bitten because USM allocators over-align in
practice. `DeviceContextCpu::allocate` at least _refuses_ over-alignment rather than lying. **Fix this before
R0, and add `DeviceContextSycl.hpp` to the blast radius.**

**R0.1's own example cannot be allocated under D5.** `HistoryBuffer` carries `alignas(kCacheLine)` on its
storage, so `alignof(HistoryBuffer<float,16>) == 64`; `DeviceContextCpu::allocate` refuses `align > 16` AND
refuses `Residency::devicePtr` outright. "Document the cap" is not available -- the allocator contract must
change.

**Three cuts adopted:**

1. **Drop the `Block<T>` base pointer.** Put `DeviceBuffer state{}` + reset flag in `DeviceBlockShadow` instead:
   the existing memcpy already carries it, no new `Block` layout, teardown rides `release()`, and the shadow's
   move ctor transfers the buffer -- which DELETES R0.3's "block move: state lost" row.
2. **Drop the opt-in re-seed fallback policy** (R0.4's last sentence): speculative, no consumer.
3. **Sequence the interior-edge device copy behind the counters** (see the D6 contradiction below).

**Corrections to the D1 half:**

- **The boundary block already exists structurally.** `makeDeviceSubGraph` inserts a `DeviceToHost` behind every
  unclaimed output (`SubGraph.hpp:208-250`). It is **per boundary port, plural**, not per chain -- so multiple
  exits are independent drains and a diamond's interior joins never see one. Contract to write down: _a drain
  owns exactly its own edge's tags; chain-level aggregation does not exist._
- **"Host-blind" can only ever mean pass-through tags.** `forwardInputTags` SUBSTITUTES the post-application
  value for setting-matching keys (`Block.hpp:1216-1231`) and rescales `sample_rate` by the resampling ratio
  (`:1166-1175`). That value exists only in host state, so setting-matching and dropped keys take the host path
  _by requirement 2_. Only the verbatim fast path (`!anyDrop && !anySubstitute`, `:1240-1243`) is eligible.
- **`gr::Message` is NOT a ValueMap** (`Message.hpp:53-89`: five `std::string` envelope fields, and an explicit
  `static_assert(!is_trivially_copyable_v<Message>)`); only its payload is. And there is **no block-to-block
  message forwarding anywhere, host included** -- messages are a scheduler bus. So "forward messages on the
  device" maps to no existing behaviour. Likely honest reading: one record FORMAT (ValueMap), two carriers --
  tags on the per-edge ring, messages on the per-block slab.
- **Fallback mid-chain needs no tag migration**: the interior tag ring is `Access::Shared`, so a CPU-fallback
  call reads and writes the same rings. Only `DeviceState` bytes migrate. This is the strongest argument for
  keeping the device-only tag-ring idea measurement-gated.

**A1 -- THIS DOCUMENT CONTRADICTED ITSELF and is now fixed:** D6 said "build it as part of R0, not deferred
behind the counters" while R0.5 said "gate it on the counters" and R0.8 step 7 said "measure, then decide D6".
Resolution pending maintainer (question 1 below); until then, treat the state half as first and the forwarding
half as second within the same milestone.

**Other ambiguities to close before implementation:** post-D3 mirror-refresh semantics (removing copy-back makes
`mirrorStateReturns` purposeless, but flipping back to epoch-keyed refresh invalidates
`qa_DeviceSpans.cpp:439`); D5's "fault loudly" holds only on backends with true device memory; `DeviceState` as
a reserved nested name (`SoapyRaiiWrapper.hpp:204` already nests one) and whether it subsumes
`DeviceStateIsReflected`; and whether a span-tier body may still mutate a reflected member after D3.

**Sharpened invariant set (a PR can be checked against these):**

1. not a setting -- snapshot identical to a twin block without the nested type;
2. **host-oracle equivalence** -- any dispatch sequence, including epoch bumps, stop/start and forced
   fallbacks, is byte-identical to the same body on `"host"` (this subsumes four of R0.9's tests);
3. no silent re-seed -- either continuity, or a terminal error;
4. single writer at a barrier, for state as well as slabs;
5. tag conservation across a chain -- per-exit drains disjoint, no duplication at fan-in;
6. (gated) zero host payload rebuilds for pass-through tags on interior edges.

**PoC correction:** run it on `host:sycl` and the CPU context too, not only `gpu:sycl` -- device-only USM there
may over-align by luck and hide both alignment defects until the framework edit.

#### R0.11 THIRD REVIEW (2026-09-02) — D2a feasibility, a live defect, and what deletes

**LIVE DEFECT, demonstrated: D7 is violated today.** `copyBackUserState` copies back EVERY reflected
trivially-copyable non-pmr member with no filter separating settings from state
(`DeviceRelocatable.hpp:158`), and `Annotated<float,"gain">` is trivially copyable. Measured: a kernel writing
`gain` persists it host-side, **settings epoch 0 -> 0, no `settingsChanged()` call**. So a device body can
change settings today, which D7 forbids. D3 (remove copy-back) closes it; R0.9 test 3 should assert exactly this
mutant.

**D2a is implementable, but NOT literally, and one gap is load-bearing.** `registerSyclRuntime()` has **no
production caller** -- verified, every call site is a test, the only other occurrence is its own definition
(`SyclRuntime.hpp:149`). Late wiring is a designed affordance today (`ExecutionStrategy.hpp:168`: "a null
resolution is never cached, so a domain wired up later still resolves next call"). **Fail-at-init would fail
every production graph**, because nothing has registered a backend by then. Needs either a
registration-before-graph-build application contract or an automatic hook -- an unmade decision.

**Refusal reasons, classified** (and the count was wrong: there are **TEN** `dispatchCpuFallback` sites, not six
-- `:176 :180 :205 :207 :233 :433 :502 :608 :695 :706`):

- init-decidable: unresolved domain, non-relocatable block, backend-serves-no-tier, mirror allocation (by making
  it eager), `canDispatch` false;
- never init-decidable, must become terminal runtime errors: **domain withdrawn mid-run**, device faults, and
  **owning-tag publication** (data-dependent -- `if constexpr` on the argument type inside the body);
- eliminable: `:695` is already dead code (guarded by the same constexpr condition at `:199`); `anyResident` was
  never a refusal reason, only a probe gate.

**Inconsistencies to fix in this document:**

- `compute_domain` is a **live reflected setting** (`Block.hpp:737`) and mid-run changes are explicitly supported
  (`:1374-1377` re-resolves) -- contradicts "blocks remain in the domain they were instantiated for". Needs a
  freeze decision.
- R0.6's "keep one test on the legacy path" is dead under D3 -- keep it as the _negative_ fixture instead.
- R0.3's domain-migration row, R0.4 in full, R0.8 step 4 and R0.9 test 5 are all dead text under fail-hard.
- D5 vs the CPU context: `DeviceContextCpu` still refuses `Residency::devicePtr` (`DeviceContext.hpp:119-121`),
  so a device-only `DeviceState` block **fails at instantiation there** -- while R0.10's PoC correction requires
  running on it. And a host-domain oracle run never reaches device dispatch, so `deviceState()` needs a
  host-side arm the design does not specify.
- D1's message half has no substrate; the clarification (messages are scheduler-dispatched) already concedes it.
  **Re-scope D1's 80% case to tags only.**

**What deletes under fail-hard** (`ExecutionStrategy::dispatch` loses its decision tree, file shrinks ~1/3):
`dispatchCpuFallback` + all ten sites; `firstFallbackWarning` and both warn-once flags; the `bulkPublishesTags`
probe, `firstUse` gating and `isFirstUseOfTheseSettings`; the `mutatesItsOwnState` family (span tier: post-D3
flip `HasDeviceProcessBulkSpans` to `const TBlock&` -- it is the only non-const tier -- and **D7 becomes
type-enforced with no probe at all**); `mirrorStateReturns`; `copyBackUserState`; `cpuFallbacksDuring` and its
20-record-ring fragility. Residue to keep: an init-time auto-parallel canary (it already uses synthetic samples)
because `mutable` still evades a const signature before C++26.

**Minimal first commit: the §80.0 transfer counters, alone.** No counter exists yet; zero behaviour change, no
dependency on any decision, and every later commit cites its numbers. **Caution:** `DeviceContext::upload/
download` is NOT the full choke point -- mirror relocation is a raw `memcpy` (`DeviceRelocatable.hpp:136-138`)
and tag staging/replay likewise, so counters placed only in the context would systematically under-report
exactly the traffic D6 must be judged on.

#### R0.12 FINAL SWEEP (2026-09-03) — D11 does NOT close the last hole; five more paths

**Verdict: NO.** The revised D11 closes whole-machine absence and _narrows_ the hatch-only hole, but five
silent-substitution or mixed-state paths remain that no decision names.

**Grammar defect in D11's own example (verified): `cpu:sycl` DOES NOT PARSE.** `ComputeDomain::mapKind` knows
only `gpu`, `fpga`, `tpu`, `host`, and an unrecognised kind maps to plain `host()`. The SYCL CPU device is
**`host:sycl`**. So the warning text must read _"'gpu:cuda' not available, functional fallback to 'host:sycl'"_
-- or `cpu` becomes a grammar alias for `host`, which is a decision, not a detail.

**P2 — an ELEVENTH fallback site, outside `ExecutionStrategy`.** `Block.hpp:1951-1962`: a device-declared block
whose _type_ offers no device path warns once and silently runs the CPU path. Reachable in shipped code:
`FFT<double>` on `gpu:sycl`, because the hatch is constrained to `float`/`complex<float>` (`fft.hpp:115`,
`:223`). The D2 conversion work-list must include it.

**P4 — silent device-INDEX substitution, no warning at all.** `longestRegisteredPrefixOf`
(`DeviceContextRegistry.hpp:41-52`) strips `:` segments until something resolves, so `gpu:sycl:3` runs on
`gpu:sycl` device 0 when index 3 does not exist. Verified. A within-device-tier silent substitution -- exactly
the class the branch forbids -- and no decision touches it.

**P5/P6 — THE LATCH IS THE LOAD-BEARING GAP.** "Decided once at start" has nowhere to live. THREE independent
per-block mechanisms silently re-consult the global registries mid-run: `ExecutionStrategy.hpp:168/173` (a null
resolution is never cached), `Block.hpp:1376` (`_deviceContext = nullptr` on ANY settings change), and
`Block.hpp:1001-1011` (`migrateFieldsToDeviceResource` re-seats pmr fields whenever a resource becomes
resolvable). Forget any one and per-block silent substitution returns: a downgraded graph silently _upgrades_
block-by-block the moment something registers a runtime mid-process, and members of one group can diverge.

**P8 — `refuseTwoDeviceDomains` is log-only at run time.** Enforced with an error at `makeSubGraph`
(`SubGraph.hpp:283-285`) but `startDispatch` merely `gr::log::error(...)` and continues (`:57-59`), so a group
whose graph was replaced via `setGraph` can run with two device domains -- breaking the single-domain invariant
per-sub-graph D11 leans on.

**D4 vs D2 vs D11 — a real contradiction.** State is "preserved across start/stop", but two starts can resolve
differently (hardware or wiring changed), and post-D2 there is NO transfer primitive to move `DeviceState`
between residencies. Preserve is then impossible and silent re-seed is forbidden. Undecided.

**Good news, verified:** the alignment items in R0.2/R0.10 are **already fixed in code** (both allocators honour
over-alignment) -- do not redo that pre-work. And the boundary question is benign: group-to-group seams are
already host seams because a SubGraph wrapper's own `compute_domain` is the default pool id, so downgrading one
group does not disturb a neighbour's edges. `HostToDevice`/`DeviceToHost` carry both a host `processBulk` and
the hatch, so they work on any target.

**Where the downgrade decision must sit (two sites, both real):** at the head of `SubGraph::startDispatch`
(`SubGraph.hpp:60`) for groups, and in `Scheduler::start` **before** `connectPendingEdges` (`Scheduler.hpp:769`)
for bare blocks and transparent groups. Miss either and edges are placed against a dead domain. A plain log line
at the decision site _is_ warn-once with no flag -- the existing per-type `static` flag
(`ExecutionStrategy.hpp:62-66`) is actively unsuitable, since two graphs in one process would share it.

**"Functional equivalent" is entirely new machinery.** `DeviceContextRegistry`'s whole public surface is
`registerContext`/`tryResolve`/`withdraw` -- there is **no enumeration API**, so "another served backend of the
same kind" cannot even be discovered today. Needs enumeration + a preference order + coherent substitution
across BOTH registries (execution _and_ edge memory), or you get mixed placement.

#### R0.13 STATUS REVIEW (independent, 2026-09-03) — corrections, and where the design is bigger than the problem

Claims re-verified against the working tree; **corrections to R0 above, which is wrong on these points:**

- **R0.0's "remove the stale `Graph.hpp:766` rationale comment" is already done** -- no such comment exists.
- **R0.10's "the shadow's move ctor transfers the buffer, DELETING R0.3's 'block move: state lost' row" is FALSE.**
  `Block(Block&&)` (`Block.hpp:889-908`) does not list `_deviceShadow`/`_deviceContext`/`_settingsEpoch`, so the
  moved-to block gets a fresh shadow. The row stands; it is harmless only because `emplaceBlockImpl` constructs
  in place and nothing moves a block after `init()`.
- **R0.10's `makeDeviceSubGraph` citation is the wrong file** -- transfer insertion is
  `blocks/basic/.../DeviceSubGraph.hpp:46-81`, not `SubGraph.hpp:208-253` (that is `boundaryPorts()`).
- **R0.11's "no transfer counter exists yet, make it commit 1" re-litigates a deletion.** The counters existed on
  this branch and were removed as dead API (d1fa1d7a, 99b0f0ef). Do not re-add them as a standalone first commit.
- **R0.12's "THREE re-consult paths" is TWO.** `ExecutionStrategy.hpp:173` re-consults only while the cache is
  null. **VERIFIED: `DeviceContextRegistry::withdraw()` has no caller anywhere in the tree**, so the
  withdrawn-domain fallback (`:178-181`) and `DeviceContext::served()` are dead and untested -- they delete at
  zero test cost under D2.
- **D15's premise is wrong (VERIFIED).** `DeviceContext::upload`/`download` are pure virtual on every backend and
  used by staging every dispatch (`DeviceContext.hpp:99-100`). D2 deleted a _policy_, not the primitives.
- **P4 is two silent substitutions, not one.** Besides `longestRegisteredPrefixOf`, `defaultSyclUsmProvider`
  maps `{kind,index}` -> `{kind,-1}` -> default resource silently (`SyclRuntime.hpp:344-370`). D14 must cover
  both, or execution and edge memory land on different devices.

**D14 NEEDS A CORRECTION -- as written it fires wrongly (VERIFIED).** `deviceIndex{-1}` means "provider default"
(`ComputeDomain.hpp:23`) and bare `"gpu"` parses to `{gpu, sycl, -1}`, a _device_ domain whose raw string is not
a registry key. Keyed on `:`-stripping, D14 would warn "'gpu' not available" **on a machine with a GPU**. Fix:
resolve the **canonical string of the parsed domain**, and warn only when a parsed index >= 0 is unregistered.
Two consequences: the message must name the canonical domain plus `ctx.name()` (the canonical `gpu:sycl` is the
_default queue's_ device, not necessarily enumeration index 0, `SyclRuntime.hpp:402-412`); and `Graph.hpp:749`
must compare **resolved** strings, because it compares verbatim today -- so `gpu:sycl` beside `gpu:sycl:0`
already yields a host seam between two blocks _on the same device_, plus a two-domain refusal.

**D15 IS VACUOUS AS THE LIFECYCLE STANDS (VERIFIED, and it is the review's strongest finding).**
`isValidTransition` permits `STOPPED -> INITIALISED` only (`LifeCycle.hpp:108`), and any `-> INITIALISED` from a
non-IDLE state invokes `reset()` (`:266-268`). So a block **cannot reach RUNNING a second time without passing
through `reset()`** -- by the state machine alone, independent of what the scheduler does. If D4's "zero on
reset" row is implemented there is no retained state on the second start, and D15 refuses nothing. One of
D4-reset / D15 is redundant.

**D8 has no mechanism to hang on (VERIFIED-ADJACENT).** `Block<T>` defines no `start`/`stop`/`reset`; the state
machine calls `Derived::` only, and `invokeLifecycleMethod` **discards the return value**, so a lifecycle method
cannot veto via `std::expected` -- only by throwing, which is banned. "Never reaches RUNNING" is therefore a NEW
mechanism. The cheap alternative already exists: dispatch `fail()` returns `ERROR` and a block returning `ERROR`
escalates the scheduler to `ERROR` (`Scheduler.hpp:938-940`) -- fail-hard at the cost of one traversal.

**Two simplifications worth taking seriously (both challenge settled decisions):**

- **S1/S2 -- let the state ride the mirror the branch already has, and drop D5 from the first cut.** The mirror is
  already a persistent per-block device allocation. Over-allocating
  `alignUp(sizeof(TBlock), alignof(DeviceState)) + sizeof(DeviceState)` and pointing at the trailing region gives
  "never re-seated by the epoch relocate" **by construction**: one allocation, one free, released exactly when a
  context changes. **D5 (device-only) is the ONLY thing forcing a second buffer**, since the trailing bytes
  inherit the mirror's residency. Requirement 4 says the state _need not_ be host-accessible, not that host
  access must fault -- and D10 already concedes D5 does not hold on `host:sycl`. The branch's own rule for tag
  rings ("device-only is an optimisation, measure first") applies here and removes the second buffer, the host
  arm, the probe exclusion and F3.
- **S3 -- D7 enforcement already runs and throws its answer away.** `bulkPublishesTags`
  (`ExecutionStrategy.hpp:295-309`) executes the real span body on a bit-copy via `mutatesItsOwnState` and does
  `std::ignore =` on the result. Using that bool _is_ D7 for the span tier, today, with no new machinery.
  **New hole D3 opens:** with copy-back gone and the mirror refreshed only on epoch, a `mutable` reflected member
  written by a const span body persists in the mirror until the next settings change, **then silently resets**.

**Sequencing is wrong-ordered (R0.8).** The state feature is _unsafe_ while any per-call fallback exists -- a
mid-run host fallback would run the host body against a device pointer. So fail-hard + the latch must come
BEFORE the core feature: (1) PoC; (2) resolver + latch + fail-hard + test migration (deletes ~1/3 of
`ExecutionStrategy.hpp`); (3) D3 + const span tier + the S3 canary; (4) DeviceState core+lifecycle merged;
(5) first consumer. Counters and slab unification are separate milestones.

**Load-bearing for the requirement:** D3, D7 (via D3), the tier restriction, and some no-per-call-fallback rule.
**Not load-bearing:** D1, D6, D11, D14, D15, D5 -- all real, all separable from R0.

**The latch, minimum edit set:** delete `Block.hpp:1376`; `ExecutionStrategy.hpp:175-181` -> `fail()` (the dead
withdraw arm simply goes); `migrateFieldsToDeviceResource` uses the resolved string instead of early-returning on
"no backend yet"; a generation counter on both registries recorded at start. Edge placement already resolves at
`connectPendingEdges` (= start) and needs only the resolved-string change. **D13 needs no per-run table** -- a
pure function `declared -> {resolvedString, DeviceContext*}` suffices, because the registries are already frozen
between start and stop (`registerSyclRuntime` is `call_once`, `withdraw` uncalled); the generation counter just
makes that explicit.

**Under-specified, will cost more later:** `deviceState()` must be callable from a const body and return
`DeviceState&`; hatch blocks run host-side so `deviceState()` there yields a device pointer (exclude or
document); the allocation alignment must be `max(alignof(TBlock), alignof(DeviceState))`; D9's veto point must be
"post-resolution" not "post-init", because a staged `compute_domain` is legal today and **is** read by edge
placement (`Graph.hpp:721-737` reads staged before active) while `refuseTwoDeviceDomains` reads active only.

**Answered, was open:** R0.0 asked whether tags on an interior device-only edge force a host round trip. **They
do** -- but per tag _event_, not per sample: samples never touch the host, while tag bytes are host-touched
three times per hop (`Block.hpp:1240-1243` forward, `ExecutionStrategy.hpp:507-534` stage). That is the argument
for keeping D6 measurement-gated.

#### R0.14 FINAL REVIEW (independent, 2026-09-03) — one live bug, one large simplification, D14/D17 contradict

**LIVE BUG, and it invalidates D18's cost analysis (VERIFIED).** `HasDeviceProcessBulkSpans` takes **`TBlock&`,
non-const** (`ExecutionStrategy.hpp:108`; `canProcessBulkDeviceSpansInvokeTest` likewise via
`std::declval<TBlock&>()`) -- non-const is what the span tier _wants_; only the view and auto-parallel tiers
require const. So `FFT`'s non-const `processBulk` IS a span-tier body and **`FFT<double>` on `gpu:sycl` is
dispatched to the device today**, bit-copied together with `_cpuFft`'s four `std::vector` members
(`algorithm/fourier/fft.hpp:94-105`, unreflected) -- a device fault on a real GPU, silently "working" on
`host:sycl`. `Block.hpp:1957` is NOT reachable for it. The transfer blocks avoid exactly this by constraining to
host spans (`TransferBlocks.hpp:27-31`); `FFT` does not. **D18's refusal message is only reachable after the
step-3 const flip**, and the earlier note here that "FFT::processBulk is non-const so FFT<double> has no device
path" was WRONG.

**THE BIG SIMPLIFICATION -- `KernelState` is NOT distinguishable from an unreflected `mutable` member, once two
small edits are made.** Precisely why an unreflected trivially-copyable member written by a span body does not
persist today: (i) `ExecutionStrategy.hpp:251` re-memcpys the host object over the mirror **every span dispatch**
(`mirrorStateReturns=true`) -- this, not the epoch, is the real reason; (ii) the same line loses it again on
every settings epoch; (iii) `copyBackUserState` is reflected-only (irrelevant under requirement 4); (iv) the
auto-parallel/view canaries `memcmp` all `sizeof(TBlock)` bytes and refuse, while the span tier runs the same
probe and discards the answer.

D3 alone kills (i) and (iii). The entire residual difference is (ii) and (iv), and both are edits inside existing
functions: make the epoch refresh copy `sizeof(Block<Derived>)` plus the **reflected** user members (the
reflection loop already exists in `copyBackUserState`), seating unreflected members once on `acquire` and never
again; and make the span-tier canary compare reflected members only, keeping the full-byte compare for the
auto-parallel tier -- which enforces the tier restriction **by construction**. Then `HistoryBuffer<float,16>`
works as an unreflected `mutable` member (trivially copyable, `HistoryBuffer.hpp:70-77`; the contracts gate only
inspects reflected members, `Block.hpp:2411-2428`), the host oracle works with **no host arm, no pointer, no
over-allocation, no probe re-aim, no alignment concern**, and D4's re-seed becomes `shadow.epoch =
kNeverRefreshed` + "full copy when never refreshed". ~20 lines of framework change against the ~150 the nested
type implies. What the nested type still adds: a `static_assert(is_trivially_copyable)` on the state, and a name
reviewers can grep.

**D14 and D17 CONTRADICT each other on `gpu:sycl:3`.** D14's example says fall back to `gpu:sycl:0`; D17's
literal ladder (canonical -> `host:sycl` -> `host`) says fall back to the SYCL CPU. The latter is unsafe: two
`gpu:sycl:3` blocks would get a **DeviceOnly** interior ring on GPU 0 (`Graph.hpp:750-751`) read by a CPU kernel
-> segfault. Resolve by defining D17's "canonical" as the **index-stripped `kind:backend`**, which makes D14's
example the ladder's first rung.

**Domain identity must be the QUEUE, not the string.** `registerContext` creates a separate `DeviceContextSycl`
per name (`SyclRuntime.hpp:158`), so `gpu:sycl` and `gpu:sycl:0` are two objects on one queue. `Graph.hpp:750`
compares strings verbatim -> a **pinned host ring between two blocks on the same GPU**, plus a two-domain
refusal. "Compare resolved strings" does NOT fix this; compare the queue, or normalise the canonical name at
registration.

**A second live defect, derived from code:** `tryResolve("gpu")` has no `:` so the prefix loop never runs -> null
-> CPU fallback, while the USM provider maps `{gpu,-1}` to GPU USM. **Two adjacent `compute_domain="gpu"` blocks
today run the CPU fallback body over a device-only ring.** D14's canonical-string resolution fixes it; it needs
a regression test.

**D8 has a free and strictly better mechanism (VERIFIED).** `checkLifecycle` tests only `REQUESTED_STOP` and
`STOPPED` (`Block.hpp:2065`, `:2073`) -- **not `ERROR`** -- so an ERROR-state block keeps working. Also
`dispatchProcessing` runs only after `workInternal` has data (`:2218-2223` returns `INSUFFICIENT_*` with no
dispatch), so **a refused block whose upstream produces nothing never dispatches and the graph completes
cleanly** -- the refusal is never surfaced. Fix both at once: resolve on the `INITIALISED -> RUNNING` transition,
set `ERROR` on failure, and have `checkLifecycle` honour `ERROR`. Decides at start, names the block at start, and
the per-`work()` cost is a compare on a value `checkLifecycle` already loads -- i.e. the measurement obligation
attached to D8 becomes moot.

**`Block<T>::changeStateTo` is the one hook point nothing hides** (~6 lines, forwarding to the state machine).
`reset()` is hidden by 25 Derived `reset()` definitions AND is called by a `reset_default` tag mid-RUNNING
(`Settings.hpp:1043-1045`, `:1204-1208`); `stateChanged()` is hidden by `BlockMerging.hpp`. Every path goes
through `changeStateTo` (`BlockModel.hpp:843`, `Block.hpp:973`, `:1489`, `SubGraph.hpp:62-63`) and no block in
the tree defines its own. Both D4's re-seed and D8's resolution hang there.

**Sequencing corrections.** Step 2's delete list must KEEP `bulkPublishesTags`, `firstUse` and the
`mutatesItsOwnState` family -- step 3 reuses them for the canary. Step 3's const flip breaks two currently-green
tests that only step 4 restores (`ZeroCrossingTriggerView`, asserted to run as a kernel at
`qa_DeviceSpans.cpp:630-637`, and `RunningTotal` at `:439`), so **steps 3 and 4 are one commit**. `Upsampler`
(`:278`) and `WeightedDifferenceSpans` (`:317`) need only a `const` added. Tests that invert to expecting ERROR:
`qa_DeviceSpans.cpp:501`, `qa_DeviceBlockStyles.cpp:248-252`.

**Also:** a KernelState-declaring functor without a persistent shadow (`!kOwnsDeviceShadow`,
`ExecutionStrategy.hpp:257`) must be a compile error -- not stated anywhere. And in a gcc-only TU of a mixed
binary the registry may report `host:sycl` served while that TU cannot dispatch at all
(`Block.hpp:1996-2000`): define **served = registry resolves it AND this TU compiled a backend**, so the ladder
collapses to `host` there.

**Stale citations corrected:** `SyclRuntime.hpp` is 208 lines (canonical registration `:162-172`, USM provider
`:97-131`, not `:344-370`/`:402-412`); `DeviceBlockShadow.hpp` is 52 lines (residency `:44`, not `:270`);
`LifeCycle.hpp` STOPPED case is `:114`; `ExecutionStrategy.hpp` fallback sites end `:608`/`:706`.

### 🔍 POST-SHIP REVIEW (independent, 2026-09-03/04) — two regressions found and fixed, three calls taken

**Both defects were mine, both inside R0's own guarantee, both now fixed with mutation-verified tests
(`a16022b6`).**

- **E1 dropped pmr container headers.** `refreshDeviceSettings` filtered on `std::is_trivially_copyable_v<F>`,
  which excludes `std::pmr::vector`. The whole-object memcpy it replaced _had_ carried the header, so this was a
  regression: a coefficient set that grew mid-run left the mirror describing storage the host had since moved or
  freed. Now filtered on `detail::isDeviceRelocatableMember`, i.e. every member the first seat carries.
  **Why the tests missed it:** both existing mid-run pmr tests change coefficients to a set of the _same size_
  (`qa_DeviceAutoParallel.cpp:147-148`, `:190-191`), where the stale header happens to describe the right storage.
- **The latch was never released, and D9 had never shipped.** `_deviceContext` was written once and never
  cleared, so a restart inherited the previous run's device even after the block was told to run on the host --
  setting said one thing, block did another, silently. Cleared on entry to INITIALISED; and a `compute_domain`
  change under a running block is now refused, which is what D9 said all along.

**A test-design lesson worth keeping.** My first version of the pmr test asserted a _one-sided_ bound
(`slopeAfter > slopeBefore * 1.5`). The defect makes the kernel read junk coefficients and the response runs
away -- measured **4751** against an expected **2.48** -- so it sailed straight through a lower bound and the
mutant survived. Rewritten against the **host oracle** (same graph, same mid-run resize, run on `host`), which
needs no tuned constants and cannot be fooled in either direction. **Prefer the oracle to a band.**

**A process lesson:** I twice read a test result from a binary whose build had failed, and once patched the
wrong one of two identical code sites because `str.replace(..., 1)` takes the first. Gate every measurement on
the build's return code; anchor every patch on text unique to the intended site.

#### Decisions taken 2026-09-04 (not pre-agreed — recording them as instructed)

**F7 — ✅ CLOSED 2026-09-04 (`4a109348`). DROP the warning and the `DeviceStateIsReflected` declaration.** It fires once per type on the pattern
that is now _recommended_, it cannot fire on wrong code, and the declaration asserts something neither the
compiler nor the framework can check (C++23 cannot enumerate the members the macro omitted). A warning with
those properties trains people to ignore warnings. Delete `DeclaresDeviceStateReflected`,
`firstUnreflectedStateWarning`, the emitting block, and the `using DeviceStateIsReflected = void;` lines in the
tests. **The hazard that CAN be detected precisely** -- the host writing an unreflected member after the first
seat, which E1 will never propagate -- is worth a Debug-only byte-range comparison later; noted, not built.

**F4 — ✅ CLOSED 2026-09-04 (`4a109348`). DELETE `withdraw()`/`served()`.** The latch made them unimplementable as written: dispatch no longer
consults `served()`, so even a real caller could not stop a running block, and the comment at
`DeviceContext.hpp:50-52` claiming it is "checked on every dispatch" is now false. Withdrawal mid-run would need
a device-lost event (SYCL exposes none portably) plus a per-work check -- a feature, not a flag. The _intent_
stays recorded here; the unreachable, untested mechanism goes.

**F5 — MAKE `runAndWait()` REPORT THE ERROR, as its own commit, measured first.** A `std::expected` returner
that reports success after the scheduler ended in ERROR is the same silent-success class this branch spent a
dozen commits removing, and it leaves the no-exceptions build uniquely blind. The blast radius is NOT "192
sites" -- it is exactly the subset whose runs already end in ERROR, each of which is a test passing while
masking one. Method: add a temporary log at the return, run both gates, count, then fix each newly-red test by
asserting what it actually meant.

### 📊 FIRST DEVICE MEASUREMENT (2026-09-04, `bm_DeviceDispatch`) — the scratch hypothesis is REFUTED, and the

### span tier is a capability, not a performance story

Machine: RTX 3070, acpp generic, Release `-O2 -march=native`, best-of-5, warm (first run JITs). Reproduce with
`./build-acpp/core/benchmarks/bm_DeviceDispatch`.

**Fixed cost of one dispatch's scratch** (9 shared-USM regions, of which two are the 64 KiB tag arenas):
`host:sycl` **3.3 us**, `gpu:sycl` **91.6 us**.

**One recursive IIR section, 65536 samples, MS/s by chunk size:**

| domain      | chunk 64 | chunk 512 | chunk 4k | chunk 64k |
| ----------- | -------- | --------- | -------- | --------- |
| `host`      | 29.6     | 38.3      | 39.3     | 39.2      |
| `host:sycl` | 1.8      | 10.9      | 28.6     | 31.5      |
| `gpu:sycl`  | 0.39     | 1.50      | 2.32     | **4.97**  |

**Cost attribution on the GPU:**

| chunk | dispatches | total   | scratch | scratch share | kernel ns/sample |
| ----- | ---------- | ------- | ------- | ------------- | ---------------- |
| 64    | 1024       | 168 ms  | 93.8 ms | **56%**       | 1133             |
| 512   | 128        | 43.7 ms | 11.7 ms | 27%           | 488              |
| 4096  | 16         | 28.2 ms | 1.5 ms  | 5%            | 409              |
| 65536 | 1          | 13.2 ms | 0.1 ms  | **1%**        | 200              |

**What this settles.**

1. **The review's hypothesis — "per-dispatch scratch dwarfs everything, hoist it and D6 gets re-sized" — is
   half right and the half that matters is wrong.** Scratch dominates at _small_ chunks (56% at 64) and is
   irrelevant at large ones (1% at 64k). Hoisting it into `DeviceBlockShadow` would buy a lot at chunk 64 and
   nothing at chunk 64k -- and at _no_ chunk size would it make this workload competitive. **Do not do it yet.**
2. **The GPU is 8x SLOWER than the host on this workload, at its best chunk size.** One GPU work item costs
   200 ns per dependent sample against a CPU core's 25.5 ns. That is not a defect: a recursive filter is
   inherently serial, the span tier runs it as **one work item**, and one GPU thread is far slower than one CPU
   thread. The span tier buys **residency** -- keeping data and state on the device between blocks -- exactly as
   the design says. It was never going to buy speed.
3. **Therefore the performance case cannot be made with the span tier.** It has to come from the
   **auto-parallel** tier (N work items over independent samples) and from _chaining_ device blocks so the data
   never returns to the host. A device IIR is the right demonstration of **capability**; an FFT, an FIR, a
   magnitude/power stage or a channeliser is the right demonstration of **performance**.
4. **`host:sycl` at 31.5 MS/s vs `host` at 39.2** is the honest cost of the dispatch machinery on the same
   hardware: about 20% at large chunks, and everything at small ones.

**THE AUTO-PARALLEL TIER, WHICH IS WHERE THE PERFORMANCE CASE ACTUALLY IS** (same machine, one dispatch per
run, 128 dependent flops/sample):

| domain      | 64k  | 256k | 1M       | 4M       |
| ----------- | ---- | ---- | -------- | -------- |
| `host`      | 21.5 | 23.3 | 25.2     | 25.4     |
| `host:sycl` | 41.6 | 79.2 | **96.5** | 47.3     |
| `gpu:sycl`  | 13.8 | 30.3 | 45.9     | **48.3** |

At 64k samples and one flop per sample the host wins everything (47.2 MS/s vs 13.9 on the GPU) -- the work does
not pay for the trip. Add arithmetic and scale the batch and both device domains overtake it:

- **`host:sycl` peaks at 3.8x the host** (96.5 vs 25.2 at 1M). SYCL on the CPU is a real win and needs no GPU --
  the most immediately useful result here for most users.
- **`gpu:sycl` crosses over at ~256k samples** (30.3 vs 23.3) and reaches **1.9x at 4M** (48.3 vs 25.4).
- **`host:sycl` falls off at 4M** (96.5 -> 47.3). Measured, not explained; suspect memory pressure on the 16 MB
  working set. Worth a look before quoting the 1M figure anywhere.

**So the demonstration set has to be built on two axes the examples must make explicit: arithmetic intensity per
sample, and batch size.** A device example that mirrors a real DSP chain -- FFT, FIR, magnitude/dB, channeliser --
at >=256k samples per dispatch shows a win; the same chain at 4k samples with one multiply shows a loss, and
showing only the first would be dishonest. Both belong in the examples.

**Consequence for the plan:** measure before hoisting anything (done -- and it stopped a premature optimisation),
and build the use-case examples on data-parallel blocks. D6 (tag forwarding) should be sized against a
data-parallel chain, where the per-event host hop is a real fraction, not against this one.

### 🎯 STANDING GUIDANCE (maintainer, 2026-09-04): FUNCTION OVER PERFORMANCE

_"First demonstrate the API (i.e. user can write C++ code that is (near-)identical for host and device
execution), then performance (cascaded block performance on device vs. on host)."_

**This reorders the remaining work and lowers the stakes on several open questions.** The demonstration set is
first about the _programming model_: the same block source, the same graph wiring, one setting changed, running
on `host`, `host:sycl` and `gpu:sycl` and giving the same answer. Only after that does the device-vs-host
throughput of a _cascade_ matter.

Consequences:

- spike A at a typical N (4096) is a legitimate **capability** demonstration even though the GPU loses there; it
  does not have to wait for the k-windows relaxation;
- the honest counter-example (small chunk, one flop -> host wins) belongs in the set as an API demonstration
  too, not only as a caveat;
- the performance story is explicitly about **cascaded blocks staying on the device**, which is the residency
  argument the IIR measurement supports, not about a single kernel beating a CPU core.

### ✅ DECISIONS (maintainer, 2026-09-04) — the work order for the demonstration set

**K1 — the k-windows relaxation lands BEFORE spike A.** An overlapped `processBulk` block may take several
frames per dispatch instead of the one `Block.hpp:1705` forces whenever a stride is active. Needs no new state;
it changes what a `processBulk` + `Stride<>` span means, and that change is accepted deliberately.

**K2 — EOS tail: `processEpilogue` is invoked IF PRESENT, and the core does nothing else.** Maintainer's words:
_"an optional, domain- and user-specified extension that is invoked if it's present, not more not less... no
padding or other data manipulations from the point of view of the GR4 core/infrastructure."_ So the framework
offers the hook and takes no view on the tail; **no zero-padding, no synthesised samples, no policy in core.**
A block that wants its tail flushed defines the hook and decides for itself what that means. The direct FIR's
dropped N+K-2 samples are then the block author's business, documented at the block, not a core behaviour.

**K3 — a window larger than its edge is a hard error at start, plus an opt-in auto-size.** Refuse with a message
naming the block and both sizes rather than hanging on `INSUFFICIENT_INPUT_ITEMS` forever, **and** give the
device scheduler/dispatcher a setting that raises the minimum edge size from the blocks' own minimum-buffer
requirements, so a user who wants it sized automatically can have that without the core guessing by default.

**K4 — the demonstration set is ONE chain, four stages, shown three ways.** `source -> FIR -> magnitude -> sink`
on `host`, `host:sycl` and `gpu:sycl`, identical block source, one setting changed, plus the cascade timing.
One story end to end; the API point lands hardest when it is literally the same code three times.

**Work order:** K1 -> K3 -> spike A (direct FIR first, then fast convolution) -> K4 chain -> cascade timing.
K2 is a check that the hook already behaves as stated, not new machinery.

#### K1 / K2 / K3 — status and the decisions I took (2026-09-04)

**K1 — ✅ LANDED as an OPT-IN mixin, not a change of default.** My first cut changed the default and broke the
`qa_Block` stride table; the correction that mattered is that **the stride table's block IS a `processBulk`
block** (`Resampler<int>`), so `processBulk` + `Stride<>` already has pinned semantics — one window per
invocation, framework advances by one stride — and K1 as originally described would have silently re-specified
them. Maintainer chose opt-in. Shipped as `gr::BatchedWindows<>` in `annotated.hpp`, declared next to
`Resampling<>` / `Stride<>`; `WindowBatchControl` in `Block.hpp` gates both the k-window early return in
`computeResampling` and the `k*S` advance in `inputSamplesToConsumeAdjustedWithStride`. Mutation-verified:
dropping the mixin from the test block turns the new `qa_Block` test red. Three configs green.

**K2 — ✅ NO WORK NEEDED, verified in code.** `processEpilogue` is already invoked under
`if constexpr (HasProcessEpilogueFunction<Derived>)` and core publishes only what the block requested. The
maintainer's specification ("invoked if present, not more not less; no padding or other data manipulation from
core's point of view") is what the code already does.

**K3 — decisions taken, splitting into two commits.** Advisor-reviewed.

- **The existing `input_chunk_size > maxSyncIn` hard error is Release-INVISIBLE.** Both call sites of
  `checkBlockParameterConsistency()` (`Block.hpp:968`, `:1379`) sit under `if constexpr (gr::meta::kDebugBuild)`.
  I am **not** un-gating it — that would newly activate every other check inside it in Release across the whole
  codebase, far beyond K3. **Open item, pre-existing, recorded here:** the resampling consistency checks do not
  run in Release builds.
- **Part 1 (hard error) goes in the `changeStateTo()` RUNNING-from-INITIALISED branch**, beside
  `decideComputeDomainForRun()`. Both `Scheduler::start()` and `::resume()` call `connectPendingEdges()` before
  that transition, so `port.bufferSize()` is final there; `applyChangedSettings` runs before connect and would
  false-error. Only **connected synchronous** ports are checked — an unconnected optional port reads capacity 0
  and must not be treated as a violation. Refusal drives the block to **ERROR**, matching the D11/D12 refusal
  semantics already shipped, not `requestStop`.
- **The boundary is the whole verification, and it is measured, not reasoned.** Ring capacity is page-rounded, so
  the test reads the actual `Edge::bufferSize()` from a probe graph and then asserts: window == capacity still
  makes progress, window == capacity+1 refuses. If equality does not progress the inequality is off by one and
  the guard-rail becomes a false refusal.
- **Part 2 (auto-size) puts the knob on `Graph`, NOT the scheduler** — a deliberate deviation from the
  maintainer's "device-scheduler/dispatcher" wording, flagged for review. `calculateStreamBufferSize` is a
  `Graph` member running at connect time, before a scheduler exists; a scheduler-owned setting cannot reach it
  without a push-down that buys nothing.
- **Part 2 must read STAGED-or-active settings.** `emplaceBlock({{"input_chunk_size", N}})` only _stages_; the
  active value is still 1 at connect time, so reading the active value alone silently no-ops. The max is taken
  over exactly three things — the edge's own `minBufferSize`, the destination's `input_chunk_size`, the source's
  `output_chunk_size` — inside the existing `forEachEdge` sweep, which fixes fan-out for free. **Known
  limitation, not chased:** the "reuse the already-connected size" early return above that sweep leaves
  later-connected edges order-dependent.
- **Part 1 ships even if Part 2 slips.** Part 1 is the guard-rail that makes spike A safe to explore; spike A at
  `N+K-1 <= 65536` never reaches the wall under the default edge size anyway.

**K3 — what the implementation actually found (2026-09-04), correcting two of the decisions above.**

- **A `Graph`-level _reflected_ setting is impossible: the top-level `Graph` never runs its own `init()`.** Only
  its child blocks do (`Graph.hpp:420,448`), so `Graph({{"auto_size_edges_to_chunks", true}})` lands in
  `_initBlockParameters` and is never applied — probed directly: staged empty, active absent, field still default.
  The knob is therefore a **plain public `bool Graph::autoSizeEdgesToChunks`**, not a reflected setting, because a
  reflected one would look settable from the ctor and YAML and silently do nothing. **Flagged for review: this is
  my call, and it is a second deviation from the maintainer's "a settings for the device-scheduler/dispatcher".**
- **It had to be added to `Graph`'s move constructor.** That ctor lists members explicitly, so the knob was
  silently dropped by `sched.exchange(std::move(flow))` — the first version of the test failed for exactly that
  reason, which is why the test asserts on `edge.bufferSize()` and not merely on "the run went green".
- **The staged-settings worry was measured and is WRONG here, so the branch was deleted.** Probed: at edge-sizing
  time `input_chunk_size`/`output_chunk_size` are **already active** (`emplaceBlock` -> `addBlock` -> `init()` ->
  `applyStagedParameters`), staged is empty. A staged fallback would have been untested dead code. **Known
  limitation, deliberate:** a chunk size applied _later_ (`settings().set()` after construction) is not seen by
  the auto-size — it then hits Part 1's hard error, which is loud rather than silent.
- **Three mutants killed, one per arm.** `>` -> `>=` turns the equality case red (the boundary is real, not
  reasoned); disabling the guard makes the over-size case **hang**, which is precisely the stall K3 exists to
  prevent and is the strongest evidence the guard is load-bearing; deleting the _output_ arm hangs the output
  case, so that arm is not dead code — worth checking because `Port::resizeBuffer` returns early for inputs and
  only the source port of each edge is ever sized.

### 🧭 §80 SPIKE READINESS — what is buildable now, what is blocked, and on what (checked in code 2026-09-04)

The maintainer's framing, recorded because the measurement reads wrong without it: **the device IIR is not a
speed claim.** Its point is that pre-processed data STAYS on the device -- filter it there, keep going there --
so the win is the device->host->device copies that never happen, not the filter itself. The 8x slower figure is
the price of one work item, and it is the right price for residency.

| spike                        | status                             | what is missing                                                                        |
| ---------------------------- | ---------------------------------- | -------------------------------------------------------------------------------------- |
| **A** FIR + fast convolution | **buildable now**                  | nothing structural -- implementation only                                              |
| **A2** IIR                   | ✅ **done** (`a6dae82d`)           | --                                                                                     |
| **B** channeliser            | **served as built** (1:1 splitter) | a _decimating polyphase_ bank still needs a window on a collection — hatch only        |
| **C** correlator             | **buildable now**                  | nothing — `3515e330` gave reductions over one span and a correlator on the window tier |
| **D** reductions / AGC       | **buildable now**                  | same commit; the window tier served this, no separate reduction tier was needed        |
| **E** rational resampler     | **buildable now**                  | nothing -- the span tier's separate in/out counts exist                                |

**A is unblocked, verified piece by piece:** `MathOpPairImpl::processOne(T, T) const noexcept`
(`Math.hpp:140`) is auto-parallel-eligible, so the fast-convolution multiply reaches a device; `gr::complex<float>`
and `<double>` are registered for all four two-port blocks (`Math.hpp:28-31`); `FFT` reaches a device through
`processBulk_sycl` (float / `complex<float>`); taps live in a `std::pmr::vector`, re-seated onto device memory;
interior device-only edges work and two spellings of one device are now one domain (F1). The premise still to
measure is **the crossover: kernel length at which frequency-domain convolution overtakes direct** -- and
`bm_DeviceDispatch` is now the instrument for it.

**B is blocked on a real framework gap.** The device tiers derive their port count from
`std::tuple_size_v<...>` over a _static_ span tuple (`ExecutionStrategy.hpp:109,113`), so a runtime channel count
(`std::vector<PortOut<T>>`) cannot reach a device at all. Two ways out, and they are different projects:
(i) a **compile-time channel count** as a template parameter -- buildable today, shows the polyphase structure
and the fan-out over one shared device ring, but not the runtime-count realism §80 asked for;
(ii) **dynamic port collections in the device tiers** -- a genuine framework feature, and the one that makes
§68.1's "interior traffic exceeds boundary traffic by the channel count" measurable.

**C and D are blocked on the same missing thing, and D existed to prove it: there is NO reduction tier.**
Confirmed by search -- `ExecutionStrategy.hpp` contains no reduction machinery of any kind. The auto-parallel
tier is strictly element-wise, the span tier is one work item, so today every reduction (RMS, power, peak-hold,
AGC, correlation peak search) must drop to the `processBulk_sycl` hatch and hand-write its own. §80's note that
this "is worth knowing now" is answered: **it is the largest remaining hole in the device programming model**,
and it is ubiquitous in real graphs.

**Suggested order, unchanged from §80 and now justified by the gaps:** A (litmus, unblocks the crossover number)
-> E (cheap, honest coverage) -> then decide B-static vs B-dynamic -> C/D once a reduction tier exists.

### ✅ K4 / SPIKE A FIRST HALF — LANDED 2026-09-04 (`06eb5f78`), and the numbers are not flattering

`core/test/qa_DeviceDspChain.cpp`: `source -> DirectFir -> Magnitude -> sink`, one block source, three
domains, **bit-identical output** on `host`, `host:sycl` and `gpu:sycl` (`std::ranges::equal`, not a tolerance).

**Choices I took, none of them pre-agreed — flagged for review:**

- block names `DirectFir` / `Magnitude`, file name `qa_DeviceDspChain.cpp`, taps `{1, -2, 0.5}`
- the host arm is checked against the **closed form** (a ramp through those taps is `y[n] = 1 - n/2`), not
  against a second copy of the same loop, so the host is not certifying itself before the devices are
  compared to it
- the timing table lives in the test rather than in a benchmark, so the demonstration is one artefact

**What was VERIFIED rather than assumed, because the design stood on it:**

- the framework really does hand a strided `processBulk` block `input.size() == frame + K - 1` while
  `output.size() == frame` — probed directly (258 / 256 for a 3-tap filter); had it handed `frame`, the body
  would have read out of bounds and _every_ arm would have been wrong in the same way, so a host-vs-device
  comparison would have passed on garbage
- the tiers are pinned by `static_assert`, not inferred: `HasDeviceProcessBulk<DirectFir<float>, float, float>`
  and `AutoParallelisable<Magnitude<float>>`. Without this a block that silently stopped qualifying would run
  on the host while its domain said otherwise, and the test would still pass

**The measurement (build-acpp, -O2, RTX 3070, 1 Mi samples, MSample/s):**

| domain    | frame 256 | frame 4k | frame 64k |
| --------- | --------- | -------- | --------- |
| host      | 49.05     | 76.77    | 88.91     |
| host:sycl | 2.61      | 9.24     | 11.64     |
| gpu:sycl  | 0.95      | 4.29     | 5.98      |

**This is the predicted result, not a regression.** The view tier is `parallelFor(ctx, 1UZ, ...)` — one work
item — so a device pays a kernel launch per frame and runs the convolution serially. Throughput rises with
frame size exactly as launch-bound behaviour does (0.95 -> 5.98 on the GPU). **The API claim is proven; the
speed claim needs the indexed tier and is the next step.** Do not quote these as "GPU DSP performance".

Debug and Release differ by ~2x on the host row (`-Og` 25 vs `-O2` 49 at frame 256) — quote Release only.

The frame-64k row needs `input_chunk_size = 65538 > 65536`, so the demonstration also exercises
`Graph::autoSizeEdgesToChunks` end to end; without it K3's hard error refuses that row.

### ✅ SPIKE A SECOND HALF — the crossover, MEASURED (2026-09-04, `a7a2ca31`)

`DirectFirSycl` is `DirectFir` plus a `processBulk_sycl` hatch: same host body, one extra method that launches
**one work item per output sample** through `gr::device::parallelFor(syclContextFor(queue), ...)` — the project's
own wrapper, so no raw SYCL in the test. Its output is asserted bit-equal to the host body before any timing is
quoted. **No framework change was needed for this**; the hatch has been there all along.

**The answer to "why DSP on a GPU", on this framework (build-acpp, -O2, RTX 3070, 1 Mi samples, frame 64k,
best of three after a warm-up, MSample/s):**

| domain / filter       | 3 taps | 32 taps | 128 taps |
| --------------------- | ------ | ------- | -------- |
| host, whole span      | 86.86  | 45.64   | 11.98    |
| host:sycl, own kernel | 73.21  | 62.19   | 32.17    |
| gpu:sycl, own kernel  | 64.22  | 59.92   | 59.05    |

**The GPU is flat in filter length and the host is not.** At three taps the run is memory-bound and the host
wins; by 128 taps the host has fallen 7x and the GPU has not moved, so the GPU is ~5.4x the host. **The crossover
is between 3 and 32 taps.** That is the whole argument, and it is now a number in a test rather than a claim.

Frame sweep at 3 taps, same conditions:

| domain / filter       | frame 256 | frame 4k | frame 64k |
| --------------------- | --------- | -------- | --------- |
| host, whole span      | 57.60     | 80.07    | 86.30     |
| host:sycl, whole span | 3.17      | 32.77    | 79.26     |
| gpu:sycl, whole span  | 0.98      | 4.47     | 6.11      |
| host:sycl, own kernel | 4.88      | 34.85    | 77.85     |
| gpu:sycl, own kernel  | 3.77      | 31.58    | 77.82     |

**Two measurement traps found and fixed in the setup, both of which had produced wrong numbers first:**

- **An edge sized to exactly one chunk starves the pipeline.** With `autoSizeEdgesToChunks` alone the GPU
  own-kernel row at frame 64k read **10.45**; with four frames of ring it reads **60-70**, a factor of five.
  No stage can start before the one ahead of it finishes when the ring holds exactly one frame.
  **✅ ANSWERED 2026-09-04 (maintainer): "at least twice the minimum required."** Shipped as
  `Graph::kChunksPerEdge = 2` (`4169d272`), mutation-tested at 1. **Two chunks recovers it in full** — the GPU
  own-kernel row at frame 64k goes 10.45 (1x) -> **77.82** (2x, graph-sized) versus 70.85 with an explicit 4x,
  so the knob alone now does the right thing and the demonstration chain no longer asks for headroom by hand.
- **AdaptiveCpp JIT-compiles on first use**, which made one row read 5.56 where it should read ~48. Every timed
  configuration now runs a small warm-up first and takes the best of three; the whole test is 11 s under -O2.

**✅ ALL SIX COMMITS VERIFIED GREEN ON THREE CONFIGS (2026-09-04):** build-gcc15-debug 103/104 (the one failure
is `qa_SoapyIntegration`, timing out with no SDR hardware attached), build-acpp **107/107**, build-acpp-registry
**147/147** — the registry config is the only one that compiles `blocks/*/test`, and it is clean.

### 📐 REVISED DEVICE-EXECUTION DESIGN — settled 2026-09-04, supersedes the "indexed tier" and "reduction tier" plans

Written after five probes and a maintainer review that corrected me on four points. **Read this before touching
the device dispatch.**

#### The programming model, in one table

| what the block writes                                       | who parallelises it                                | when it applies                                     |
| ----------------------------------------------------------- | -------------------------------------------------- | --------------------------------------------------- |
| `const noexcept processOne(T...)`                           | framework, one work item per sample                | element-wise, no neighbours                         |
| `const processBulk(span, span)` + `Resampling<>`/`Stride<>` | **framework, one work item per window** (to build) | anything with a fixed window -> fixed output count  |
| an `algorithm/` type owning its own `q.submit`              | the algorithm itself                               | cooperation, multi-kernel plans, library transforms |
| `processBulk_sycl(queue, spans...)`                         | the block                                          | ONLY to reach an `algorithm/` type from a block     |

**`processAt` is dropped.** The framework already computes the work items — see the probe table below.

#### 1. Parallel `processBulk` — the whole of the "indexed tier", with no new entry point

Proven, same body, only the K1 opt-in differing:

| declaration                                  | calls    | in.size() | out.size() | closed form |
| -------------------------------------------- | -------- | --------- | ---------- | ----------- |
| `Resampling<K,1>+Stride<1>+BatchedWindows<>` | 1        | 4096      | 4094       | ✓           |
| `Resampling<K,1>+Stride<1>`                  | **4094** | **3**     | **1**      | ✓           |

The framework **already invokes the FIR body once per output sample with a K-sample sliding window**. Those 4094
invocations _are_ the work items; they are merely serial. The tier = launch them concurrently.
`k = resampledOut / output_chunk_size`, both already at the dispatch site -> **no plumbing**.

**Two properties of the scheme, to be stated rather than fixed:** a parallel `processBulk` cannot publish tags
(no ordering across work items); a slice needs `streamIndex + offset` for absolute position (both spans already
carry `streamIndex`). **And a fixed per-window output count is what slicing requires** — which is why variable-rate
blocks (below) cannot be sliced.

#### 2. Cooperation belongs in `algorithm/`, NOT in a framework tier — I was wrong twice

- **WRONG:** "you cannot launch a nested parallel region from inside a kernel, so cooperation needs a tier."
  True but irrelevant. SYCL 2020 has no device-side enqueue and **does not need one**: cooperation is ONE
  `nd_range` launch plus `sycl::group_barrier`, exactly as the maintainer said.
- **WRONG:** "the reduction tier needs new `nd_range` machinery in `DeviceContextSycl`." **The machinery is
  already in this repo** — `SyclFFT.hpp:329-365` uses `sycl::nd_range`, `sycl::local_accessor` and
  `sycl::group_barrier` today. It simply is not behind `DeviceContextSycl::parallelFor` (a flat `range<1>`).

**The model to copy is `SyclFFT`:** the algorithm takes a `sycl::queue&`, owns a multi-kernel plan, and chains
stages with `sycl::event` + `h.depends_on(dep)` (`submitVanLoanStage` -> `submitLocalStages`) instead of waiting
between them. A reduction and a scan belong in the same bucket, with host and device backends, reused by any
block. **`DeviceContextSycl::parallelFor` staying a flat `range<1>` is then fine** — cooperation never goes
through it.

#### 3. The FFT: the maintainer's account is what the code does — my earlier explanation was a strawman

`SyclFFT` is **Van Loan Stockham auto-sort, radix-2, chosen explicitly for having no bit-reversal** — more
arithmetic, but no data-dependent permutation and therefore no swap/sync — while the host production path is a
different algorithm (`SimdFFT`, butterfly with bit-reversal). My "one work item per output would be O(N^2)"
answer described something nobody proposed. **The real reason the FFT belongs on the hatch is that the algorithm
owns an event-chained multi-kernel plan**, which no per-output tier can express. Same for every library transform.

#### 4. Static port collections (spike B): two small changes, not a project

- The type system **already models them**: `stream_output_port_types<Splitter<4>>::at<0>` is `std::array<float,4>`.
  My earlier "collections collapse to a span-of-spans" claim was wrong.
- Measured: a collection block **runs correctly on `host`** and lands in **ERROR on `host:sycl` and `gpu:sycl`** —
  the D2a "no device path for these types" arm, because no tier concept matches `processBulk(TIn&, std::span<TOut>&)`.
- `processOne` returning the collection value **does not compile**: `Block.hpp:1832` does
  `output_range[i] = result` where `output_range[i]` is a whole `OutputSpan`. For a collection the loop must be
  `output_range[c][i] = result[c]`.
- **The maintainer's unification is right:** a single port is a collection of size one, so ONE signature shape
  suffices. Probed: a single port cannot take the collection signature today
  (`HasProcessBulkFunction = false`). Two options — rewrite every block to the uniform shape (not lean), or have
  the **device tier concepts accept the collection shape so single ports match by wrapping** (lean). **Take the
  second.**
- **Per-channel state is a CONTRACT, not a detail.** A 10-channel IIR bank needs one state per channel, and
  correctness depends on the framework guaranteeing **one work item per channel** — work item `c` touches only
  `state[c]`. If dispatch ever splits a channel across work items, the state races silently. Same shape for
  multi-channel AGC, per-antenna delay lines, per-carrier equalisers.

#### 5. Recursive and variable-rate blocks

- **Recursive (IIR, AGC loop) still belongs on the device.** They do not parallelise, but they run there through
  the span tier with `mutable` state, and **residency is the point** — no device->host->device hop in a long
  chain. My earlier "stays serial by construction" wording was misleading and is withdrawn.
- **Variable-rate IS parallelisable, in two phases.** A piecewise-constant ratio gives closed-form output
  positions -> embarrassingly parallel. A truly per-sample-varying ratio is a sequential recurrence, i.e. a
  **prefix sum**, parallelisable as a scan (log N, two passes) followed by a parallel gather. So it needs a scan
  primitive — the same `algorithm/` bucket as the reduction. It still cannot be _sliced_, because slicing needs a
  fixed per-window output count.

#### 5b. Host/device detection inside an algorithm — the `constexpr` ask is NOT satisfiable under SSCP

The maintainer asked for a `constexpr`/`consteval` boolean so an `algorithm/` type can pick its host or device
implementation inside the processing function. **Measured in the AdaptiveCpp headers this build uses:**

- This build compiles `--acpp-targets=generic`, i.e. **SSCP**, which is a _unified host-device pass_: one
  compilation serves both, so no C++ constant expression can distinguish them.
- The discriminator is `HIPSYCL_SSCP_STAGE1_IR_CONST int __acpp_sscp_is_host` (`s1_ir_constants.hpp:26`) — an
  **IR constant filled in by the JIT at stage 1**, used through a plain `if`, not `if constexpr`
  (`backend.hpp:145`: `__acpp_backend_switch` is `if (__acpp_sscp_is_host) ... else ...`).
- So the honest shape is an `inline bool` (say `gr::device::isHostExecution()`) that the JIT constant-folds when
  it specialises — **zero runtime cost, dead branch eliminated, but not `constexpr`**.
- `__acpp_if_target_device` IS compile-time, but only in non-SSCP explicit-target builds, so it is not portable
  here. `AtomicRef.hpp:29` already records that using it interferes with SSCP kernel metadata.
- **There is no CPU-vs-GPU discriminator inside a kernel at all** — SSCP exposes exactly two IR constants,
  `__acpp_sscp_is_host` and `__acpp_sscp_is_device`. The maintainer's "2nd function" for SYCL-CPU vs SYCL-GPU
  therefore cannot live in the kernel: that choice must be made **host-side, where the queue is**, which is
  exactly what an `algorithm/` type does (it holds the queue and can query the device).
- `constexpr bool gr::device::kHasSycl` already exists for the separate question "was this build compiled with
  SYCL at all", which is the host-only fallback switch.

#### 5c. `BatchedWindows<>` DOES duplicate existing semantics — the maintainer's challenge is upheld

Probed directly (`Resampling<>` block, 4096 samples, counting invocations):

| declaration                                                     | calls    | max in.size() | max out.size() |
| --------------------------------------------------------------- | -------- | ------------- | -------------- |
| `Resampling<8,8>`, no `Stride<>`                                | 1        | 4096          | 4096           |
| `Resampling<8,8>+Stride<>`, stride = 8 (== In)                  | 1        | 4096          | 4096           |
| `Resampling<8,8>+Stride<>`, stride = 0                          | 1        | 4096          | 4096           |
| **`Resampling<8,4>+Stride<>`, stride = 4 (< In, real overlap)** | **1023** | **8**         | **4**          |
| same + `BatchedWindows<>`                                       | 1        | 4096          | 4092           |

**A resampling body is ALREADY required to loop over a span holding many chunks in four cases out of five.**
The overlap case is the lone exception, created by `Block.hpp:1727` ("with stride, we cannot process more than
one chunk"). So `BatchedWindows<>` adds no new information about the block — **it removes an exception**, and as
a separate top-level mixin it is redundant vocabulary.

**Blast radius of simply removing the exception is small — only two shipped blocks declare `Stride<>`:**
`fourier::FFT` (whose body already loops, `fft.hpp:95` `for (b = 0; b < nBatches; ++b)`) and
`FrequencyEstimatorFrequencyDomainDecimating` (`Stride<0U>`, back-to-back, so unaffected). The real cost is the
`qa_Block` stride table, whose `exp_counter`/`exp_in`/`exp_out` values encode the anomaly.

**Three options, for the maintainer:** (a) keep the mixin as shipped — no behaviour change, redundant vocabulary;
(b) **delete the mixin and remove the exception**, so overlap behaves like every other resampling case — leanest;
(c) keep the relaxation but spell it as a parameter of `Stride<>`, where the anomaly actually lives, instead of a
new top-level concept.

**(b) MEASURED, not estimated (2026-09-04).** Gates removed, batched windows made the default:

| `n_samples=1000, chunk=100, stride=50` | (a) as shipped | (b) default |
| -------------------------------------- | -------------- | ----------- |
| `processBulk` calls                    | 19             | **1**       |
| `in.size()` per call                   | 100            | 1000        |
| `out.size()` per call                  | 100            | 1900        |
| total input the block SEES             | 1900           | **1000**    |
| **total output produced**              | **1900**       | **1900**    |
| samples consumed from the edge         | 1000           | 1000        |

**Output and consumption are identical; only the presentation changes.** Under (a) the overlapped region is handed
to the block 19 times, so it sees 1900 samples of a 1000-sample stream. Under (b) it sees each sample once and
derives the 19 windows itself.

**Tag boundaries already bound the batch:** `computeSampleLimits` clamps `availableToProcess` by `nextTagLimit`,
so a batched span never spans a tag. That answers the first objection a reviewer will raise against (b).

**Blast radius of (b) in the whole repo: 17 assertions in the `qa_Block` stride table, plus the `fourier::FFT`
body (see below — my first report that the FFT was unaffected was wrong).**
All 17 are in the four _overlap_ cases (stride < chunk); every non-overlap case passes unchanged. **`fourier::FFT`
— the only shipped block using real overlap — is completely clean**: `qa_FFT` 904 + 2697 asserts and
`qa_FFTDevice` 2590 asserts across 54 tests all pass, because its body already loops over `nBatches`.

The remaining differences are the ones worth weighing: dispatch count 19 -> 1 (on a device, 19 kernel launches
versus one — this is the 5.8 vs 78 MSample/s gap); bodies must loop over `out.size()` (already true in 4 of 5
resampling cases); a block summing `input.size()` reads 1000 instead of 1900 (arguably the honest number, but a
change); output arrives in one burst rather than 19, bounded by `requestedWork`.

#### 5c-RESOLVED. (b) CHOSEN AND IMPLEMENTED (maintainer, 2026-09-04) — and it is a 20x performance change, not a cleanup

`BatchedWindows<>` is **removed** (reverting that part of `dcd72265`), the `nResamplingChunks = 1` exception is
**deleted**, and an overlapping strided block now behaves like every other resampling block: it is handed as many
windows as the ports allow. **Net negative API surface — one concept and one exception gone, nothing added.**

**The measurement that justifies it far beyond tidiness.** Same demonstration chain, same blocks, before/after:

| domain / filter       | frame 256 (a) | frame 256 (b) | change  |
| --------------------- | ------------- | ------------- | ------- |
| host, whole span      | 49.28         | 67.29         | 1.4x    |
| host:sycl, whole span | 2.81          | **58.87**     | **21x** |
| gpu:sycl, whole span  | 0.97          | **5.88**      | **6x**  |
| host:sycl, own kernel | 2.59          | **69.74**     | **27x** |
| gpu:sycl, own kernel  | 3.30          | **58.86**     | **18x** |

**Throughput is now essentially flat in frame size** (256 / 4k / 64k all within ~15%), where before it rose
steeply with frame — i.e. **the chain was launch-bound and is no longer.** The one row that does not move is
`gpu:sycl, whole span`, now a constant 5.88 at every frame: a whole-span body is one work item, so batching buys
it one launch instead of 4096 but the single work item still does all the arithmetic. That row behaving exactly
as predicted is the check that the rest of the table is real.

**Test-side consequences, all resolved rather than blessed:**

- The four _overlap_ cases in the `qa_Block` stride table change; every non-overlap case is untouched. Each new
  value was verified against the derived invariant `total_out == nWindows * output_chunk_size` with
  `nWindows = 1 + (n_samples - input_chunk_size)/stride`, **not** blessed from a measurement.
- That invariant is now **asserted in the harness**, so the table is anchored by an oracle that no batching
  decision can move.
- `Resampler` now records **the windows its span holds** rather than the raw span, so both
  `exp_in_vector` expectations survive **unchanged** — proving the framework still hands a span from which
  exactly the declared windows are derivable.
- **`fourier::FFT` DID need a change, and my earlier "it is clean" claim was WRONG.** Its body computed
  `nBatches = available / N` and took frame `b` at input offset `b * N` — correct for back-to-back batches,
  wrong once the span holds _overlapping_ windows. Under (b) the strided FFT test failed
  `[64 == 208]`: with 13 windows of 16 at hop 4 it produced 4 back-to-back frames instead of 13 overlapping
  ones. **Fixed**: both the host and the `processBulk_sycl` path now walk the span by
  `hop = stride == 0 ? N : stride`, take `nBatches = min(1 + (in.size() - N)/hop, out.size()/N)`, and read frame
  `b` at `b * hop` while writing it at `b * N`. The SYCL path keeps its single bulk `memcpy` when `hop == N` and
  gathers frame-by-frame only when the frames overlap.

  **Why I got this wrong, and it is a repeat of §77.4.** The earlier check greped the test output for
  `"tests failed|all tests passed"`. Boost.UT prints a failing suite as `tests: 2 | 1 failed` — which matches
  **neither** pattern. The filter could only express good news, so a real failure read as silence. _Any grep over
  test output must match the failure signatures too, not just the success ones._ The registry config is what
  caught it, which is also why that config exists.

#### 5d. Host/device detection: the compute-domain route beats the SSCP one (maintainer, 2026-09-04)

The maintainer's suggestion — a method plus the compute-domain / device-context info — is **strictly better than
`__acpp_sscp_is_host`**, and the pieces already exist:

- `ComputeDomain::parse()` is **`constexpr`** and yields `kind` ("host"/"gpu"/...), `backend` ("sycl"/...) and
  `isDevice()` (`ComputeDomain.hpp:31`).
- `Block` already caches it: `bool _computeDomainIsDevice` (`Block.hpp:802`), set in `cacheComputeDomainKind()`.
  A plain `bool`, so it **already rides the device mirror and is readable inside a kernel**.
- The resolved context carries the rest: `DeviceContext::backend()`, `deviceType()`, `isGpu()`
  (`DeviceContext.hpp:50-55`).

**This gives what SSCP cannot: SSCP exposes only `is_host`/`is_device` and has NO CPU-vs-GPU discriminator.** The
compute-domain route separates `host`, `host:sycl` and `gpu:sycl` — the "2nd function" the maintainer asked for.

**The two-level pattern already exists in shipped code** — `SyclFFT::forward` (`SyclFFT.hpp:150-163`):

```cpp
#if GR_DEVICE_HAS_SYCL_IMPL          // level 1: compilation gate, host-only build never sees SYCL types
    if (auto* sycl = syclCtx(ctx)) { // level 2: runtime, from the DeviceContext
        stockhamGpu(...); return;    //          device algorithm (Stockham, no bit-reversal)
    }
#endif
    _simdFft.compute(...);           // host algorithm (SIMD butterfly), the default
```

**Why `#if` and not `if constexpr (kHasSycl)`:** `if constexpr` still requires the discarded branch to _parse_,
and in a host-only build the SYCL types do not exist. So `#if` is necessary wherever SYCL types are named; a
`constexpr bool kHasSycl` is the right spelling only for user-level algorithm code that does not name them.

Caveat: `compute_domain` is a runtime setting, so the method is `constexpr`-callable but not a compile-time
constant — a cheap uniform branch, not `if constexpr`. In practice free, because the CPU-vs-GPU choice must be
made **host-side before submitting** anyway (only the host holds the queue). Layering:
`constexpr kHasSycl` (built with SYCL at all) -> compute domain / `deviceType()` (which device, host-side at
submit) -> the kernel body needs no branch.

#### 10. ✅ SPIKE D — per-frame reduction (2026-09-05, `95e1c847`)

`FrameRms` in `core/test/qa_DeviceDspChain.cpp`: `Resampling<N,1> + Stride<N>`, many samples in, one figure
out, repeated per frame. **No new mechanism** — a segmented reduction is a window like any other, so the
existing slice runs each frame as its own work item. 463 asserts / 6 tests on acpp, 121 on gcc15; the host arm
is checked against the closed form for a ramp _before_ being used as the oracle for the device arms.

**Deliberately NOT promoted to a public block.** A per-frame RMS is three lines of arithmetic; a separate
`algorithm/` header for it would be the abstraction §8.1 warns against, and the point of the spike is the
_tier_, not the maths. `blocks/electrical/PowerEstimators.hpp` already computes RMS for its own domain (and is
itself a `Resampling<100,1>` block — the same shape, blocked from a device only by its dynamic port
collections, i.e. spike B).

**Cooperative reduction of a SINGLE large frame remains out of this tier** and belongs to an `algorithm/` type
that submits its own `nd_range`, per §2 above.

#### 11. ✅ SPIKE B — fixed-width port collections on a device (2026-09-06, `ae45ea35`)

**Three fixes, all smaller than the "dynamic port collections" project I first scoped:**

1. **`DeviceRelocatable` refused the port array as unrelocatable state** — that was the actual blocker
   (`member 'out' cannot be relocated`). A collection of ports is still ports; a kernel touches none of them.
2. **The host per-sample loop assigned a whole channel span** where it meant one sample of one channel
   (`Block.hpp:1832`) — it did not compile, so it had never run.
3. **Device staging assumed one buffer per port.** A collection now gets ONE channel-major allocation
   (`nChannels * count`), so it still costs one buffer and one copy-back whatever its width.

**The channel count comes from the block's port declarations, not the spans** — the runtime span tuple has
already erased it. `stream_output_port_types<TBlock>::at<kIdx>` is `std::array<float,4>` for a 4-channel port.

**Two traps found while wiring it:**

- Naming `stream_output_port_types<TBlock>` in a member alias breaks for **plain functors** (the device test
  helpers), which carry no port descriptors. Guarding with `if constexpr` is not enough — the alias is checked
  when the class is instantiated. It needs a namespace-scope specialisation that yields `void` instead.
- **`gr::complex<float>` has a `std::tuple_size`**, so a complex-valued port was misread as a 2-channel
  collection. The predicate must also require the port value to be a _range_ over its channels.

#### 12. ✅ SPIKES C, A-fast-convolution AND E2 — the list is complete (2026-09-06)

**C — correlator + `algorithm/Reduce.hpp`** (`8f8686ff`). The per-lag arm is a window and needed nothing new.
The peak search does: reducing ONE span means work items cooperating on one answer, and only whoever submits
the kernel can ask for a work-group launch — so `Reduce` sits in `algorithm/` with `nd_range` +
`group_barrier`, the `SyclFFT` pattern, and a block reaches it through the hatch.

**The finding worth keeping:** the device correlator disagreed with the host in the fifth digit, and against a
double-precision reference **the device was the MORE accurate of the two** (2.5e-4 vs 3.5e-4) — a device may
contract `a*b+c` into one rounding. The test now holds both arms to the exact correlation instead of to each
other. _Bit-equality is the right assertion for a pointwise chain and the wrong one for anything that sums._

**A fast convolution** (`79d70f89`). `algorithm/filter/FastConvolution.hpp` + `blocks/filter/`, overlap-save,
transform supplied by the caller so one routine serves a host and a device transform. Declares the same window
shape as the direct filter, so the two are interchangeable in a graph. **Trap it caught:** assigning `taps`
directly bypasses `settingsChanged`, so the block kept its default single unit tap and passed the signal
through — the test read as a filter bug and was a settings-plumbing bug.

**E2 drift resampler** (`5c1af806`). The cubic-Hermite kernel lifted out of `DriftCompensator`, where it was
reachable only at an insert/drop boundary, into `algorithm/filter/HermiteResampler.hpp` + a block. It is the
counterpart to the rational resampler and shows the cost of an arbitrary ratio: **no fixed window means no
per-window work item**, so it owns its accounting and stays sequential. **Livelock found and fixed:** at EOS a
window shorter than the four-point interpolant produced nothing and consumed nothing, spinning forever; the
tail is now drained, and `in.min_samples` keeps that from happening mid-stream.

**State at 16 commits, all three configs 100%** — gcc15 107/107, acpp 111/111, registry 154/154
(`qa_SoapyIntegration` excluded: it stalls on its timeout with no radio attached).

#### 13. ✅ THE THREE MAINTAINER DECISIONS, IMPLEMENTED (2026-09-06)

All three of the calls listed as "awaiting the maintainer" were answered and are now in.

**Edge sizing is a reflected setting** (`9f8a1cae`). `Graph::auto_size_edges_to_chunks` is an
`Annotated<bool>` inside `GR_MAKE_REFLECTABLE(Graph, ...)`, asked for like any other setting:
`gr::Graph flow({{"auto_size_edges_to_chunks", true}})`. **The one thing that made this awkward** is that a
graph is the only block nobody else initialises — nothing calls `settings().init()` on it — so it applies its
own settings once, at the top of `connectPendingEdges()`, which is the first point that needs them. _I took a
narrower route than the option's wording ("init the top-level Graph" from the scheduler): a lifecycle change
has unbounded blast radius, and the observable behaviour asked for is the same either way._

**The generally useful spike blocks were promoted** (`6067741b`). `blocks/filter/Correlator.hpp` and
`blocks/filter/FrameStatistics.hpp` (`RootMeanSquare`) are public, registered blocks with their own tests
against closed-form oracles. `DirectFir`, `Magnitude` and `Channeliser` stay in `qa_DeviceDspChain`, where
being deliberately naive is the point.

**The fast convolution reached the device** (this entry). `FastConvolutionFilter::processBulk_sycl` drives
`gr::device::SyclFFT`: widen real to complex, `forwardBatch`, multiply by the tap spectrum, `inverseBatch`,
take the useful tail — three small kernels around two batched transforms, every frame in the span at once.

**Two things worth keeping from doing it:**

- **The hatch bypasses the window slice.** `dispatchSyclBulk` is checked _before_ the window arm, so a block
  that declares a window and also owns a hatch gets the whole batched span in the hatch and must walk the
  frames itself by its own hop. That is exactly the FFT defect fixed in `faa442d2`; it is a property of the
  dispatch order, not of either block, and anything adding a hatch to a windowed block will meet it.
- **Scratch is allocated per dispatch**, matching what `ExecutionStrategy`'s staging path already does. Caching
  it in the block would need a hand-written move constructor to avoid a double free, which is the price
  `SyclFFT` pays; at these frame sizes it did not show up against the transform.

`FastConvolutionFilter<double>` has no device arm — `SyclFFT` is float-only, so the hatch constrains itself
away and double precision stays on the host. Missing capability, not a failure.

#### 14. THE FILTER-LENGTH SWEEP IS NOW SHARED AND GOES TO 65536 (maintainer, 2026-09-06)

Every table that varies filter length uses the same eight lengths — **16, 32, 64, 512, 1024, 8192, 32768,
65536** — held in one place, `gr::test::kFilterLengths` in `core/test/device_test_helpers.hpp`, which
`blocks/filter/test` now reaches through one `target_include_directories`. Below 16 and above 65536 a FIR
comparison measures memory layout rather than arithmetic, so the sweep stops there deliberately. The exact
crossover point is not the deliverable; the orders of magnitude are.

**Two things the sweep forced, both of which the fixed-`N` tables had hidden:**

- **Each length gets the window it deserves** (`max(4096, bit_ceil(nTaps))`). A fixed 4096-sample window at
  65536 taps makes overlap-save transform 131072 points to keep 4096 of them — a 32x waste that would have
  read as the algorithm losing.
- **The two arms need different run lengths.** A direct filter costs one multiply-add per tap, so a fixed
  sample count spends the whole test in the longest filter; a transform-based arm costs the same per sample
  whatever the filter length, so shrinking _its_ run only measures the fixed cost of building a graph. Scaling
  both together produced a non-monotonic overlap column (27 at 512 taps, 6.8 at 8192, back up to 19 at 65536)
  that was pure per-run overhead. Each arm now picks its own: the direct arm scales, the transform arm always
  streams `kStreamSamples`.

**The number the branch was missing** (MSample/s, `qa_FastConvolutionFilter`, measured idle):

| taps  | direct/host | overlap/host | direct/gpu | overlap/gpu |
| ----- | ----------- | ------------ | ---------- | ----------- |
| 16    | **100.51**  | 32.90        | 20.29      | 92.49       |
| 64    | 27.45       | 33.66        | 8.85       | **93.51**   |
| 1024  | 1.27        | 29.88        | 0.39       | **85.01**   |
| 65536 | 0.02        | 9.62         | 0.00       | **49.03**   |

Algorithm and domain are independent choices and they multiply. On the host, overlap-save turns a 5000x fall
into a 3.4x one, crossing over between 32 and 64 taps. On the GPU the direct form is the _worst_ column at
every length — a window at one flop per tap cannot pay for a dispatch — while the transform is exactly the
shape a batched launch wants. Changing only the domain makes things worse; changing only the algorithm buys
9.62; both together buy 49.03, which is 2450x the direct host filter it replaces.

The direct-FIR chain in `qa_DeviceDspChain` tells the same story without the algorithm axis: the device
advantage is 1.4x at 16 taps and 330x at 65536, because over the sweep the host falls 2500x and the GPU 11x.

#### 16. 🔧 THE THREE-PHASE REORDER — plan, defect log, and the trap in it (2026-09-06)

**What the maintainer asked for.** One branch, three phases: first every commit that fixes an issue _already
present in `origin/main`_ (to be cherry-picked into a separate branch later), then the device-enabling work in
**didactic** order, then the spikes as demonstration. Commit timestamps preserved. Every fix for a defect this
branch itself introduced is folded into the commit that introduced it, and **what went wrong is recorded here
instead**, because after the fold the commits no longer show it.

**Mechanism.** Scripted `cherry-pick` onto a fresh branch off `origin/main` — this environment blocks `rebase -i`.
Author dates survive a cherry-pick by default; committer dates need `--committer-date-is-author-date` or they all
collapse to today. A squash via `cherry-pick -n` + `commit --amend` **silently resets the author date to now**
unless `--date=` passes the original back, so every target's author date is captured before the rewrite starts.

**Phase 1 — measured, not guessed.** Each candidate was cherry-picked onto a throwaway worktree at bare
`origin/main`. Ten apply cleanly and are phase 1; four do not and are therefore _not_ phase 1 whatever their
subject lines say:

| commit                                                                | applies to bare main?                       |
| --------------------------------------------------------------------- | ------------------------------------------- |
| `fe58a4c7` release both halves of a double-mapped buffer              | yes                                         |
| `09be6bc4` keep a moved block's memory resource                       | yes                                         |
| `ee68f90b` the header `std::ignore` comes from                        | yes                                         |
| `c098319e` stop three tests failing on things they do not test        | yes                                         |
| `5da25aa0` keep the HTTP client out of every core consumer            | yes                                         |
| `91f36b7e` let a wire map nest another without allocating             | yes                                         |
| `b8e4703f` read a wire map's values without materialising one         | yes                                         |
| `bdba3190` publish a tag without a concept enumerating the map types  | yes                                         |
| `b0a6f47f` let a history buffer keep the memory resource it was given | yes                                         |
| `ff4d1d5a` keep a choice of mode as the enumeration it is             | **no** — `fft.hpp` conflicts                |
| `c4d0727b` a stride now overlaps for a block that consumes for itself | **no** — `fft.hpp`, `Block.hpp`             |
| `6825af7b` the chain A/B was measuring a defect                       | **no** — `qa_FFTPerformance.cpp`            |
| `87323984` the two-port form of the arithmetic blocks                 | **split** — feature yes, its device test no |

**⚠ THE TRAP: blame-origin is not conceptual origin, and folding by blame alone produces commits that cannot
build.** Blaming the exact lines each fix changes gives a _lower bound_ on where it can go, never the answer.
`45e7cbfb` ("a tier that keeps the block's ratio must not have its counts collapsed") blames entirely to
`1c712857`, the commit that first wrote those `ExecutionStrategy.hpp` lines — but it fixes the **window slice**,
which does not exist until `8681f400`, thirty commits later. Folding it where blame points would place a fix for
a mechanism before the mechanism exists. The rule actually used is therefore: **fold into the later of (a) the
newest branch commit any changed line blames to, and (b) the commit that introduced the mechanism the fix's
subject names.**

A second trap sits underneath: **fixes chain**. `22e2499f` blames 43 lines to `c1eaae30`, which is itself a fix
being folded away. Targets must be resolved transitively, or a fold lands on a commit that no longer exists.

**Defect log — what each folded fix was fixing.** After the reorder these commits are gone from the history;
this table is the only remaining record of what went wrong and why.

| folded fix | into                    | what was actually wrong                                                           |
| ---------- | ----------------------- | --------------------------------------------------------------------------------- |
| `0e19ee61` | `0e2c6e3b`              | a device buffer ignored the alignment it was asked for                            |
| `8257bd99` | `e9868df7`              | the single-device-domain rule was checked at construction, not on a running group |
| `c1eaae30` | `1c712857`              | nothing warned when a block kept state the device could not see                   |
| `1c858e91` | `835ee8e8`              | a queue's device context could die while the queue was still in use               |
| `b18f6b02` | `e9868df7`              | a group spanning two devices reported the problem and then started anyway         |
| `4614615e` | `835ee8e8`              | the FFT's host engine was reachable from device code                              |
| `6b7b633e` | `6629e69d`              | the copy-back became dead once the mirror kept state, and stayed                  |
| `4855e034` | `fe63a3c5`              | two spellings of one device read as a domain boundary to the graph                |
| `29cc3726` | `6629e69d` + `94815118` | the mirror missed a resized setting; a decided domain could still move            |
| `22e2499f` | `94815118`              | the start-time decision and the latch left machinery unreachable                  |
| `c4d0727b` | `d1a7701c`              | a stride did not overlap for a block that consumes for itself                     |
| `af7cbbb2` | `79b98bd5`              | an auto-sized edge held one chunk where it needed two                             |
| `260af43e` | `d1a7701c`              | the FFT ignored the stride it declared when walking a batched span                |
| `45e7cbfb` | `8681f400`              | a tier that keeps the block's ratio had its counts collapsed to their minimum     |
| `6825af7b` | `835ee8e8`              | the chain A/B was measuring a defect and reporting it as a design choice          |

**Style and docs.** The three `docs(device)` commits fold cleanly into the feature each documents. The two style
sweeps do not: `1ead6d86` alone changes ~500 lines across 36 files whose origins span 15+ commits. Hunks whose
file has one obvious origin are folded; the cross-cutting remainder stays as one small `style:` commit rather
than being attributed by guesswork.

**Verification: three points only** — the phase-1 tip, the phase-2 tip and the final tip, on all three configs.
Every intermediate commit is therefore **unverified by construction**. That is normal for a reordered branch and
is stated rather than implied.

#### 20. 📐 R1 — superseded, see `port_collection_intent.md`

The design once written out here was reviewed on 2026-09-08 and its premise did not survive: the device tiers
do have static collection support, and no in-tree block needs a dynamic one. The plan, its findings and its
costings live in `~/gr4-snapshots/port_collection_intent.md` — the single source of truth for port collections
and the embedded footprint work. Nothing about the topic should be duplicated back into this file.

#### 19. ✅ CLEANUP COMPLETE — the branch is review-ready (2026-09-07)

**Verified on every configuration, at both boundaries that matter:**

|                                                | GCC 15    | AdaptiveCpp | acpp + registry     |
| ---------------------------------------------- | --------- | ----------- | ------------------- |
| phase-1 tip (9 commits, on bare `origin/main`) | 89/89 ✓   | 88/88 ✓     | not run (low value) |
| final tip (56 commits)                         | 107/107 ✓ | 111/111 ✓   | 156/156 ✓           |

Zero build errors anywhere, zero formatter violations, all 56 author dates equal to their committer dates.

`pr_message.md` rewritten against the current branch: the three-phase structure with phase 1 listed for
cherry-picking, a _What the demonstrations show_ section carrying the filter-length and direct-vs-overlap-save
tables, and the limitations updated (fixed-width port collections, float-only device transform, Debug-only
resampling checks).

**The style-sweep fold is dropped**, with reason: the lint pass rewrote many of the same lines, later commits
genuinely build on the sweeps' text (`fe63a3c5` will not apply without `1ead6d86`), and the payoff was cosmetic.

**Next, in the maintainer's order:** a thorough design for R1 before any R1 code, then R1 (dynamic-width port
collections), R2 (double-precision device transform), R3 (un-gate the Release resampling checks).

#### 18. 🔍 THE LINT PASS AND WHAT AN INDEPENDENT REVIEW FOUND (2026-09-07)

`syclExperiments` is **56 commits**: phase 1 is 9, phase 2 is 35, phase 3 is 11 (the last being a `style:`
sweep). All author and committer dates aligned. Green on all three configurations at the tip — gcc15 107/107,
acpp 111/111, registry 156/156, zero errors.

**A fresh-context review of the ten lift-to-main commits found things this session had missed.** It was run
with no knowledge of how the commits came about, which is why it beat the advisor: the advisor inherits the
framing that produced the mistake.

| finding                                                                                                                | verdict                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| ---------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `get_if<T>` type-confuses tensors                                                                                      | **REAL, HIGH.** A tensor entry records its _element_ type in `valueType` (`ValueMap.hpp:275`), and `valueEntry()` rejected only unset entries — so `get_if<float>` on a `Tensor<float>` returned a live pointer into the tensor's own header. Fixed by `scalarEntry()`, which excludes tensor and nested-map entries; the `ValueMapView` branch still needs plain `valueEntry`, so a blanket rejection would have broken nested-map reads. Regression test **mutation-proved**: red with `valueEntry`, green with `scalarEntry`. |
| the `<tuple>` commit is a no-op                                                                                        | **REAL, and my error.** `origin/main` already includes `<tuple>` in all six files; the commit added a _second, mis-ordered_ copy to five of them. I had verified `std::ignore` was used and never that the include was missing, then defended the commit on that basis. Dropped.                                                                                                                                                                                                                                                 |
| `readUriToString` inline mismatch                                                                                      | **REAL.** Declared non-inline (`PluginLoader.hpp:67`), defined `inline` (`PluginLoader.cpp:9`) — IFNDR, and a Release link failure waiting for the optimiser to inline every call.                                                                                                                                                                                                                                                                                                                                               |
| method-level `@brief` ×4, `Complex.hpp` dated audit trail, `GR_PMT_EXTERN_TEMPLATE` leak, empty `namespace gr::pmt {}` | real, all fixed                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| `DeviceStateIsReflected = void;`                                                                                       | real — device vocabulary with no definition or consumer on main, invisible at HEAD because a later commit deletes it, but present in the _lifted_ commit                                                                                                                                                                                                                                                                                                                                                                         |
| CMake `ENVIRONMENT` clobber                                                                                            | real, but **mis-attributed**: it came from `3ba501d7` in phase 3, not `866a1b51`. `qa_DeviceDspChain` was overwriting the sanitizer suppressions `add_ut_test` sets; now `APPEND ENVIRONMENT_MODIFICATION`.                                                                                                                                                                                                                                                                                                                      |

**⚠ I FELL INTO THE TRAP THE `restyle` SKILL DOCUMENTS.** I classified four CMake files as "pre-existing drift
in `origin/main`" by checking copies under `/tmp` — where `cmake-format` cannot find `.cmake-format.yaml` and
therefore reports _everything_ as mis-formatted. Re-checked in place: **all five are conformant on main**, so
every CMake drift is branch-introduced. Only `ci.yml` genuinely predates the branch. The planned phase-1
formatter commit was built on that false reading and became one `style:` commit at the tail instead.
**Never check formatter conformance on a copy outside the repository.**

**What the replay taught about folding fixes:**

- A patch cannot be folded by context into a commit whose file the _later_ commits re-indent. `11cf1835-b`
  failed for exactly that reason, and the semantic edit had to be made by hand. When `f8d3ed33` then
  conflicted, the resolution was to take its re-indentation _and_ keep `scalarEntry` — the review's
  "this hunk belongs in the earlier commit" finding showing up as a mechanical fact.
- A doc-policy fix nearly got stranded in the tail `style:` commit, which would have meant lifting phase 1 to
  main **carrying the violation**. Anything phase 1 must not ship has to be folded into phase 1, not swept up
  at the end.

**Phase 1 caveats for whoever lifts it:** `b0a6f47f` (history buffer keeps its resource) is pre-existing-issue
material but sits in phase 2, because its test needs `gr::PmrMigratable` from the device work; its test would
need adapting to travel. `c4d0727b` was checked against the phase-1 tip directly and conflicts, so it is _not_
phase-1 material.

#### 17. 📋 DECISIONS AND THE WORK QUEUE AFTER THE REORDER (maintainer, 2026-09-06)

**Phase 1 stays where it is, and it is a COLLECTION POINT.** Not lifted to its own branch yet, deliberately:
further refactoring is expected to turn up more defects that pre-exist in `origin/main`, and those belong at the
front of the branch with the other ten. Anything found from here that is not caused by this branch goes into
phase 1.

**R4 — hatch versus window slice: RESOLVED, keep the dispatch order.** Reordering so the slice wins would call
a `processBulk_sycl` once per window, i.e. one kernel launch per frame — exactly the cost the batched span
exists to remove (~20x at small frames, measured) — and it would break both existing hatches. The order is
semantically right: the hatch means _the block owns the launch_, so it must receive the raw span.

_The actual defect is not the order, it is that the frame arithmetic is hand-rolled._ The FFT and the fast
convolution each independently wrote `hop = stride ? stride : N; nWindows = min(1 + (n - N)/hop, out/N)`, and
the FFT got it wrong — that is what `260af43e` fixed. Two copies, one bug: the signature of a missing helper.
**The helper already exists** as `ExecutionStrategy<TBlock>::windowGeometry`, private to the framework tier.
The fix is to promote it to a free function in `gr::device`, have the strategy call it, convert both hatches to
it, and document it as the supported way for a hatch to find its frames. No new concept for a block writer —
it is the window they already declared with `Resampling<>`/`Stride<>`. ~2 h.

**Work queue, in the order the maintainer set:**

1. **Lint the whole branch** — GR4 style guide (§1 naming, §2 member order, §3 doc policy), `clang-format-18`,
   and a pass to reduce accidental complexity. Defects found that pre-exist in `origin/main` go into phase 1.
2. **Thorough design for R1** before any R1 code.
3. **R1** dynamic-width port collections, **R2** double-precision device transform, **R3** un-gate the
   Release-invisible resampling checks.

**Two loose ends approved for one more history pass:** reword `ee68f90b` (its message describes work upstream
subsumed; it now only adds six `#include <tuple>` lines), and attempt the blame-split fold of the two style
sweeps into what they clean up. _Sequencing note: the lint pass in item 1 will rewrite many of the same lines,
so folding the style sweeps first and re-linting them after is duplicated work — the reword is cheap and can go
now, the style fold is better done after the lint pass, or dropped if the lint absorbs it._

#### 16.2 ✅ THE REORDER IS LANDED AND VERIFIED (2026-09-06)

`syclExperiments` now **is** the reordered history: **55 commits — 10 phase 1, 35 phase 2, 10 phase 3**, on
`origin/main` at `1e145087`. `git diff` against `backup/syclExperiments-post-rebase-20260906` is empty, so the
content is provably unchanged. Author date == committer date on all 55.

**Verification, stated as it is rather than as it would look best:**

| tip                  | gcc15     | acpp      | registry                              |
| -------------------- | --------- | --------- | ------------------------------------- |
| phase 1 (10 commits) | 89/89 ✓   | 88/88 ✓   | **not verified**                      |
| phase 2 (+35)        | 102/102 ✓ | 106/106 ✓ | **not verified**                      |
| final (+10)          | ✓         | ✓         | 156/156 ✓ (fresh build dir, 0 errors) |

**Why the two registry cells are empty, and it was self-inflicted.** Wiping `build-acpp-registry/plugins`
before reconfiguring corrupts the build directory past what a reconfigure can repair — 25 `CMake Error`s out of
`GnuRadioBlockLibMacros.cmake:239`, because the generator does not re-emit sources it believes already exist.
The build then never completed and **ctest silently ran the stale binaries from the previous commit**, which is
why phase 1 reported 146 tests when it has far fewer, and why `install_consumer_smoke` "failed". Those numbers
described nothing. Recovering them needs a fresh registry build directory per tip, ~2 h, for little information:
gcc15 and acpp exercise the ordering at both tips and the final state is green on all three.

**Rule for next time: never delete generated sources from a configured build directory. Recreate the directory
or leave it alone.** And a build whose `rc != 0` invalidates the ctest run that follows it — check the build
result before reading any test count.

**`c4d0727b` resolved — it is NOT phase-1 material.** Its `Block.hpp` half was tested against the phase-1 tip
directly and conflicts there, so it depends on device-branch changes to `Block.hpp`; only its `fft.hpp` half was
ever suspected. It stays where it is, as its own commit ahead of the window group.

**What phase 1 is for.** The ten commits stand alone on `origin/main` — they build and test there — and are
ready to cherry-pick into a separate branch. One caveat when that happens: `b0a6f47f` (history buffer keeps its
memory resource) had to move **out** of phase 1 into phase 2, because its test uses `gr::PmrMigratable`, which
the device work introduces. The fix itself is pre-existing-issue material; only its test is not portable.

#### 16.1 ✅ THE REORDER IS BUILT — 71 commits became 55 (2026-09-06)

**`reorder/syclExperiments`, 55 commits: 11 phase 1, 34 phase 2, 10 phase 3.** Author _and_ committer dates
match on all 55, so `git log --date=short` shows the original timestamps either way.

**The proof that nothing was lost: `git diff reorder/syclExperiments backup/syclExperiments-post-rebase-20260906`
is EMPTY.** Identical trees across all 129 files. This is the check that matters — every fold, split and
collapse preserved content exactly — and it also means the final tip needs no rebuild, because an identical tree
builds identically to the already-verified HEAD. Only the phase-1 and phase-2 tips were built.

**Two dead-ends removed, both confirmed by measurement rather than opinion.** Blaming HEAD back to each commit
shows how much of what it added still survives:

- `a31a0683` **23% alive** — it adds `BatchedWindows<>` (21 lines in `annotated.hpp`) and `4b8e3d0f` deletes all 21. The opt-in the maintainer rejected. The pair is replaced by one commit carrying their net effect, removing
  ~152 lines of add-then-delete churn.
- `c1eaae30` **26% alive** — the state canary, mostly removed again by `22e2499f`. Same treatment.
- `060591cf` **38% alive** — docs rewritten by `c43ab415`; the two are one commit now.

**Four corrections the execution forced on the plan in §16 — the table there was wrong on all four:**

1. `260af43e` cannot fold into `d1a7701c`; the FFT stride defect only exists once a block is handed _batched_
   windows, so its origin is the overlapping-window commit, ~15 commits later.
2. `22e2499f` cannot fold into `94815118`; it removes machinery the _state_ work made unreachable, so it folds
   after `4be0f6e3`.
3. `29cc3726` needs both state tests present, so it cannot fold into the feature as a whole. **It is split by
   nature instead** — framework halves into `4be0f6e3`, test halves into the test commit. Folding it whole put
   two framework fixes inside a `test(device):` commit, which is precisely the shape this reorder exists to
   remove.
4. `c4d0727b` is not foldable at all: `8f29ed9f` needs its `Block.hpp` stride change, which sits before the
   window group. It stands as its own commit. **Open question worth checking: its `Block.hpp`/`CircularBuffer.hpp`
   half may fix a defect pre-existing in `origin/main`, in which case it belongs in phase 1.** Only its
   `fft.hpp` half blocked the phase-1 apply test.

**The general lesson, which cost four cycles before it was learned:** blame gives a lower bound on where a fix
can go, and the _file overlap_ of unapplied commits gives the real constraint. `git log --reverse origin/main..<fix>^ -- <files>`
answers "what does this still need" directly, and asking it first is far cheaper than discovering it as a conflict.

**Where the fold rule had to yield.** Five commits straddle the phase boundary — their framework half belongs to
phase 2 and their test half to phase 3 (`8681f400`, `45e7cbfb`, `8d8d02d9`, `af7cbbb2`, `b8133491`). Each was
split by path rather than forced into one phase. The two style sweeps were **not** folded into what they clean
up: later commits build on their text (`fe63a3c5` fails to apply without `1ead6d86`), so they keep their
original relative position. That is a deviation from the instruction, taken because the alternative does not
apply.

**Base state before the rewrite:** rebased onto `origin/main` at `1e145087`, 71 commits, all three configs green
(gcc15 107/107, acpp 111/111, registry 156/156). Backups at `backup/syclExperiments-pre-rebase-20260906` and
`backup/syclExperiments-post-rebase-20260906`.

#### 15. 📌 EVERYTHING THAT REMAINS (2026-09-06) — the authoritative open list for this branch

**No spike work is left.** All eight are implemented, measured and green on three configurations
(gcc15 107/107, acpp 111/111, registry 156/156). What follows is everything else, each with why it was left,
what it would cost, and what it depends on. Nothing here blocks the branch from being reviewed.

**R1 · Port collections and embedded footprint** — see `~/gr4-snapshots/port_collection_intent.md`.
Re-scoped 2026-09-08: dynamic width is not the item worth building. Deferred to its own PR.

**Status 2026-09-09.** 24 commits over `origin/main`, grouped by functionality, every commit formatted
(C++, CMake, markdown) and build-clean in sequence; tree identical to the 53-commit history it replaced. All six
DSP spikes served. R4 was already done in `USER_API_GPU_Blocks.md`; D7's premise was stale (the span body is
const) and only a residual canary remains.

**Fallback policy — SETTLED and implemented.** A compute domain is a preference; a trailing `!` (`gpu:sycl!`)
makes it a requirement. Without it every route back to the host — device absent, or no device path for the
block's types — warns once per run and runs on the host; with it each stops the graph. The marker is part of the
domain spelling, never reaches the registry, and needs no graph or global property.

**R2 · A double-precision device transform** — ~2 days.
`gr::device::SyclFFT` is float-only (`using C = gr::complex<float>` throughout, and its twiddle tables with it),
so `FFT<double>` and `FastConvolutionFilter<double>` constrain their hatches away and stay on the host. This is
a **missing capability, not a defect**: every host arm is correct and every test passes. Cost is a templated or
duplicated Stockham tier plus twiddle tables; the block-side plumbing is one relaxed `requires`. _Worth doing
only if a double-precision device chain is actually wanted — say so before it is built._

**R3 · `Block.hpp`'s resampling consistency checks are `kDebugBuild`-only** — ~half a day, plus fallout.
Found while building the resampler spikes and **deliberately not widened**. The checks that would have caught the
declared-ratio collapse do not run in Release. Un-gating them is one line, but it newly activates _every other_
check in that block in Release, and what those turn up is unknown until tried. Pre-existing, not caused by this
branch. _Do this on its own, never inside a feature commit._

**R4 · The dispatch order puts the hatch before the window slice** — no work, a documented property.
`dispatchSyclBulk` is checked before the window arm, so a block that both declares a window and owns a
`processBulk_sycl` receives the **whole batched span** in the hatch and must walk its own frames by its own hop.
Two blocks have now met this (`faa442d2` in the FFT, and the fast convolution written that way from the start).
It is correct as designed — the hatch is the escape from framework accounting — but it is a trap for the next
author. _Either leave it and document it in `USER_API_GPU_Blocks.md`, or make the slice reachable from inside
the hatch. Not a decision I should take alone._

**R5 · Scratch inside a hatch is allocated per dispatch** — no work unless it measures.
`FastConvolutionFilter::processBulk_sycl` allocates and frees its complex work buffer every call, matching what
`ExecutionStrategy`'s staging path already does. Caching it in the block would need a hand-written move
constructor to avoid a double free on a moved-from `DeviceBuffer` — the price `SyclFFT` pays for the same thing.
At the frame sizes measured it does not show against the transform. _Revisit only with a measurement showing it._

**R6 · PR structure — THE ONE OPEN QUESTION FOR THE MAINTAINER.**
Four framework changes are independently useful and independently reviewable — batched windows as the default,
the parallel window slice, `DispatchOutcome::honoursDeclaredRatio`, and port collections counted as relocatable.
The spike blocks, the promoted `Correlator`/`RootMeanSquare`, and the tests are purely additive on top. **One PR
or two (framework first, demonstrations second)?** This shapes `pr_message.md` and restructuring later costs more
than deciding now.

**Process traps re-confirmed this round, both of which look like real failures and are not:**

- **The registry config needs a reconfigure after blocks are added to a block-library directory.** Otherwise
  `libGrFilterBlocksShared.so` fails to link against `gr_blocklib_init_unit_*` symbols the generator has not
  listed. `cmake -S . -B build-acpp-registry` fixes it; the second build was 0 errors.
- **A benchmark sweep must scale each arm's run length by that arm's own cost.** Scaling both arms by the
  direct filter's cost produced a non-monotonic overlap-save column that was pure per-run graph overhead, not
  algorithm behaviour, and would have been reported as a finding.

### 📋 SPIKE STATUS AT 13 COMMITS (2026-09-05)

| spike                                | state   | evidence                                                                      |
| ------------------------------------ | ------- | ----------------------------------------------------------------------------- |
| **A** direct FIR + crossover         | ✅ done | `qa_DeviceDspChain`, bit-equal 3 domains; GPU flat 3→128 taps vs host 7x fall |
| **A2** IIR                           | ✅ done | `qa_DeviceSpans`, `a6dae82d`                                                  |
| **D** per-frame reduction            | ✅ done | `95e1c847`                                                                    |
| **E** rational L/M resampler         | ✅ done | `fd2407e0` + `86228a68`                                                       |
| **E2** cubic-Hermite drift resampler | ✅ done | `5c1af806`                                                                    |
| **B** channeliser                    | ✅ done | `ae45ea35`, 4 channels identical on 3 domains                                 |
| **C** correlator                     | ✅ done | `8f8686ff`, with `algorithm/Reduce.hpp`                                       |
| **A** fast convolution               | ✅ done | `79d70f89`                                                                    |

**Framework work that came out of the spikes, all landed:** batched windows by default (`1e57b271`), the FFT
walking its span by stride (`faa442d2`), the parallel window slice (`ca720a31`), and the declared-ratio fix
(`86228a68`).

#### 9. ✅ SPIKE E — rational L/M resampler (2026-09-05), and the two defects it exposed

`algorithm/.../filter/PolyphaseResampler.hpp` + `blocks/filter/.../RationalResampler.hpp`, tests in
`algorithm/test/qa_PolyphaseResampler.cpp` (13 asserts, oracle-based, mutation-verified) and
`blocks/filter/test/qa_RationalResampler.cpp` (41 asserts, 4 tests, bit-equal host / host:sycl / gpu:sycl).

**The block body is four lines and needs no window arithmetic:**

```cpp
for (std::size_t n = 0UZ; n < output.size(); ++n) {
    output[n] = Algorithm::sampleAt(window, taps, _phaseLength, interpolation, decimation, n);
}
```

It is correct whether the framework hands it ONE window (sliced, on a device) or _k_ windows (batched, on the
host), because `floor(n*M/L)` is already global: output `n = w*L + r` maps to input `w*M + floor(r*M/L)`.
**Second block to confirm the property the whole scheme rests on.**

**DEFECT 1 (framework, fixed): `Block.hpp` collapsed both sample counts after a framework-managed device
dispatch.** `const std::size_t count = std::min(processedIn, processedOut); processedIn = processedOut = count;`
with the comment "the framework tiers are 1:1". The slice made that false. Symptom: the resampler produced 2048
samples on a device where the host produced 3060 — **values bit-identical, only the count wrong**, which is the
shape a reviewer would misread as a correctness bug. Fixed by `DispatchOutcome::honoursDeclaredRatio`, set by the
sliced path, which suppresses the collapse. **Anything added to the framework tiers with a non-1:1 ratio must set
this, or its output is silently truncated to its input count.**

**DEFECT 2 (algorithm, fixed): each polyphase phase must sum to 1 on its own.** A unit-sum _prototype_ does not
imply unit-sum _phases_, and since one output takes exactly one phase, unequal phase sums beat at the output
rate. Symptom on a ramp: consecutive output steps of -0.80, +2.16, +0.65, -0.84 where 2/3 was expected. After
per-phase normalisation: 0.645..0.677. `decompose` now normalises each phase; the algorithm test asserts every
phase sums to 1, and its oracle reconstructs the prototype **from the phases** so it tests the indexing rather
than re-deriving the taps.

**Test-design note:** the ramp-slope assertion was rewritten to what a 24-tap prototype actually guarantees — the
_average_ slope is exact (1e-3), individual steps ripple because each phase is a different, non-ideal fractional
delay. Asserting the tight per-step value was my error, not the filter's.

#### 8. ✅ THE PARALLEL SLICE IS IMPLEMENTED (2026-09-05) — a plain `processBulk` now runs its windows at once

**No new entry point, no new user vocabulary.** A block that declares `Resampling<>`/`Stride<>` and whose body
compiles against plain spans gets one device work item per declared window.

**Mechanism** (`ExecutionStrategy.hpp`): `windowGeometry(block, nIn, nOut)` derives
`hop = stride == 0 ? input_chunk_size : stride`, `nWindows = nOut / output_chunk_size`, and refuses to decompose
unless `nWindows >= 2 && (nWindows-1)*hop + input_chunk_size <= nIn`. `runDeviceBulkCore` then launches
`nWindows` items instead of `1UZ`, work item `w` seeing `in[w*hop, +input_chunk_size)` and
`out[w*output_chunk_size, +output_chunk_size)`. A new arm ahead of the span tier routes such blocks to it.

**Why matching the plain-span signature is the right permission to slice:** a body that compiles against
`std::span` cannot call `consume`/`publish`, so it does no accounting of its own and has no per-invocation state
to race. The concept _is_ the proof.

**Two defects fixed on the way, both latent before slicing:**

- `dispatchDeviceBulk` staged **both** directions with `min(nIn, nOut)`, so a decimating window
  (`input_chunk_size > output_chunk_size`) got an under-sized input span and read past its end. Each direction is
  now staged with its own count. Not previously observable because generic bodies took the span tier.
- `invokeBulkViews` passed span **temporaries**, which cannot bind to a body taking `InputViewLike auto&`. Both
  entry points now materialise named locals.

**Measured, same chain, ordinary `processBulk` body (no hatch), MSample/s:**

| domain, whole span   | before slice | after slice |
| -------------------- | ------------ | ----------- |
| host:sycl, frame 256 | 61.8         | **81.5**    |
| gpu:sycl, frame 256  | 5.8          | **36.5**    |
| gpu:sycl, frame 4k   | 5.7          | **27.2**    |
| gpu:sycl, frame 64k  | 5.8          | 6.0         |

**The frame dependence has inverted, and that is the check the numbers are real.** Parallelism is `nOut /
output_chunk_size`, so a _smaller_ frame now means _more_ windows per span and more work items; at frame 64k the
edge holds only ~2 chunks, so `k ~ 2` and there is nothing to spread. Where the old behaviour was
launch-bound at small frames, the new one is parallel there and serial at large ones — the opposite failure,
and a much more useful one, since a large frame can always be split.

All 448 asserts still pass, including the bit-exact host-vs-device comparison and the closed-form oracle — which
is what proves the **output offsets** are right, the one thing that would otherwise produce plausible wrong data.

#### 7. WHERE THE PARALLEL SLICE GOES — located in code 2026-09-05, with a correction to my own test

**The slice belongs in the SPAN tier (`dispatchDeviceBulkSpans`), not the view tier.** Probed: a generic
`processBulk(InputViewLike auto&, OutputViewLike auto&) const` satisfies **both** `HasDeviceProcessBulk` (view)
and `HasDeviceProcessBulkSpans` (span), and `dispatch` checks the span tier **first**
(`ExecutionStrategy.hpp:172`). So every block written the generic way — including `DirectFir` — already lands on
the span tier.

That is also the tier that _can_ be sliced: `dispatch` passes `nIn, nOut` **separately** to it, whereas the view
tier collapses them to `count = std::min(nIn, nOut)` (`:153`). Measured through the data path, a
`Resampling<258,256>+Stride<256>` block receives `in.size() = 3842`, `out.size() = 3840` — i.e.
`(k-1)*S + In` and `k*Out` for k = 15 windows, exactly right.

**Both device bulk tiers are `parallelFor(ctx, 1UZ, ...)`** (`:509` span, `:645` view) — one work item. The slice
replaces that `1UZ` with `k = nOut / output_chunk_size`, giving work item `w` the sub-spans
`in[w*stride, w*stride + input_chunk_size)` and `out[w*output_chunk_size, (w+1)*output_chunk_size)`.
**The output offset is the thing to verify, not just the size** — a body writing `out[n]` into a wrongly-offset
slice produces plausible-looking wrong data, and `qa_DeviceDspChain`'s closed-form oracle is what catches it.

**CORRECTION to my own test:** the `static_assert` in `qa_DeviceDspChain.cpp` names `HasDeviceProcessBulk` (view
tier). It is true but does not pin what is exercised — `DirectFir` runs on the span tier. Same defect class as
"a test asserts something it does not exercise". To be fixed with the slice work.

**Latent, narrow, not live:** the view tier hands `min(nIn, nOut)` to _both_ spans, so a body taking concrete
`std::span` parameters with `input_chunk_size > output_chunk_size` would receive an under-sized input span and
read past its end. Nothing in the repo has that shape (generic bodies all take the span tier), but the view tier
is documented "1:1 by construction" while `HasDeviceProcessBulkForSpans` does not enforce 1:1.

#### 6. The hatch is rare and stays rare

Three users in the whole repo, two of them infrastructure: `HostToDevice`/`DeviceToHost` (a cross-residency
`memcpy` has no per-sample shape) and `FFT` (multi-kernel plan). **A user writing ordinary DSP should never need
it** — which is exactly what the parallel-`processBulk` scheme buys.

### 🔬 THE INDEXED TIER IS RETRACTED — the framework already computes the work items (2026-09-04)

**Maintainer's proposal, and it is right:** a `const processBulk` block declared with `Resampling<>`/`Stride<>` already
states its window->output mapping, so `work()` can dispatch the k windows in parallel and keep the existing
signature. **No new entry point. `processAt` is dropped.**

**PROVEN by probe, same body, only the K1 opt-in differing:**

| declaration                                  | calls    | in.size() | out.size() | closed form |
| -------------------------------------------- | -------- | --------- | ---------- | ----------- |
| `Resampling<K,1>+Stride<1>+BatchedWindows<>` | 1        | 4096      | 4094       | ✓           |
| `Resampling<K,1>+Stride<1>` (no opt-in)      | **4094** | **3**     | **1**      | ✓           |

The framework **already calls the FIR body once per output sample with a K-sample sliding window**. Those 4094
invocations _are_ the work items; they are merely run serially. The parallel tier = launch them concurrently.
`k = resampledOut / output_chunk_size` — both already at the dispatch site, so **no plumbing is needed**;
`computeResampling` computes `nWindows` and discards it, but it is recoverable for free.

My own FIR was declared `Resampling<N+K-1,N>+Stride<N>` — "one big window" — which is why it could only ever be
one work item. Declared the maintainer's way it decomposes for free.

**Mapping of every spike onto processOne/processBulk:** A direct FIR = `Resampling<K,1>+Stride<1>` parallel bulk;
A multiply = auto-parallel `processOne`; A FFT = hatch (permanent); A2 IIR = serial _on the device_ (residency is
the point); B channeliser = collections, see below; C per-lag = parallel bulk, C peak search = reduction;
D per-frame RMS = `Resampling<N,1>+Stride<N>` parallel bulk, D AGC feedback = serial; E rational = `Resampling<M,L>`
parallel bulk, E drift = variable-rate, serial.

**Two properties to state, not fix:** a parallel `processBulk` cannot publish tags (no ordering across work items);
a slice needs `streamIndex + offset` for absolute position (both spans already carry `streamIndex`).

### 🔬 STATIC PORT COLLECTIONS — smaller than I claimed; TWO corrections to my own record

**Correction 1: the type system already models them correctly.** `stream_output_port_types<Block>::at<0>` for
`std::array<PortOut<float>,4>` is `std::array<float,4>` — the per-sample value across channels. My earlier claim
that "collections collapse to a span-of-spans so PortValue deduces to a span" was **wrong**.

**Correction 2: they are REFUSED on device, not broken.** Measured: a `const processBulk(TIn&, std::span<TOut>&)`
splitter runs correctly on `host` (state STOPPED, 64 samples/channel, right values) and lands in **ERROR on both
`host:sycl` and `gpu:sycl`** — the D2a "no device path for these types" arm, because no tier concept matches the
span-of-spans shape.

**And `processOne` returning the collection value does not compile** — `Block.hpp:1832`:
`output_range[i] = result` where `output_range[i]` is a whole `OutputSpan`, not a sample. For a collection the
loop must index `output_range[c][i] = result[c]`. **So B is not a project:** (a) teach `invokeProcessOnePure` and
`runAutoParallelCore` to index collections per channel, (b) add a tier concept accepting the collection shape.
The maintainer's NTTP plan (1/4/10, later dynamic) is the right shape and mostly already supported.

**Still not done, and the reason:** the indexed parallel tier (A1). The hatch gets the performance but the block
carries a second method, so it is not the "identical source" claim. The tier would give a per-output-sample body
_through the framework_, with no SYCL in the block at all. It is a genuine framework addition -- a new block
entry point on both the host and device paths -- and therefore the next thing to put to the maintainer rather
than to build unilaterally.

### 🚧 SPIKE A — DESIGN ROUND (advisor, 2026-09-04) + the decisions I took

**Confirmed: today every non-pointwise kernel must be hand-written in the `processBulk_sycl` hatch.** Verified in
code: auto-parallel launches `parallelFor(ctx, count, ...)` and **discards the index one line before calling the
body** (`ExecutionStrategy.hpp:657-658`); `can_processOne` probes with port _value_ types, so no overload can
receive it. The view tier (`:651`) and the span tier (`:517`) are both `parallelFor(ctx, 1UZ, ...)` — one work
item. `DeviceInputSpan::streamIndex` exists and is correctly filled but only the span tier constructs one, and
that tier is serial. **The span tier cannot simply be launched with N work items**: every item would run the
whole body, `publishTag` does a read-modify-write on one shared `_acct->tagsPublished` (`DeviceSpans.hpp:153`),
and `mutable` state is by definition one delay line. Do not go there.

**THE THING I MISSED, and it changes the shape of A more than the tier question does.** With a stride active a
block processes **exactly one chunk per `work()`** (`Block.hpp:1705-1708`, pinned by `qa_Block.cpp:809`), and the
framework then consumes `stride` regardless of what the body consumed (`finaliseIO`, `:2189`). So an overlap-save
chain built on the strided FFT hands **one N-sample frame per dispatch** to everything downstream — and my own
measurement says the GPU only overtakes the host at >=256k samples per dispatch. **The chain is launch-bound by
construction unless N itself is large.** Survivable for a crossover-in-K measurement at fixed N = 2^18..2^20
(`fft_size` caps at 1048576, `fft.hpp:67`), but it must be _stated_ as a structural property of the design, not
discovered as a surprise in the numbers.

**FFT + stride IS UNTESTED. `d544a0b8` is a one-line change and `grep stride blocks/fourier/test/*` returns
NOTHING (verified: 0 references).** Everything in spike A stands on it. **First commit is therefore a host test
that `FFT(N, stride=S)` emits N per S consumed with the K-1 overlap intact, then the same on the device.**

#### Decisions taken (mine, not pre-agreed — flagged for review)

**A1 — ⚠️ CORRECTED 2026-09-04: the indexed tier is the PERFORMANCE step, not a prerequisite.** The maintainer's
standing guidance settles the order: _"function over performance. First demonstrate the API (i.e. user can write
C++ code that is (near-)identical for host and device execution), then performance."_ A direct FIR written as a
**view-tier** `const processBulk(std::span<const T>, std::span<T>)` with `Resampling<N+K-1, N>` + `Stride<N>` and
pmr taps is **identical source on `host`, `host:sycl` and `gpu:sycl` with zero framework work** — that is the K4
deliverable, buildable today. §80's litmus survives intact: the view tier is framework-dispatched and the taps are
a seated `std::pmr::vector` member, so "taps read from a framework kernel" still holds. What the reorder gives up
is only the parallel-speedup number — on the GPU the view tier is one work item, so **the device timing for the
direct FIR will be slow, and that is expected, not a failure.** The indexed tier below is what buys the speed, and
it comes after the chain runs. Spike A order is now: view-tier direct FIR -> fast convolution -> indexed tier.

**A1 (original, superseded) — build the INDEXED PARALLEL TIER first, then spike A on it.** Ranked above the hatch because the crossover
has two sides and _today the direct-FIR side has no parallel tier either_, so a hatch-only A would compare a
hand-written frequency path against a serial time path and measure nothing about the framework. It also keeps
§80's litmus intact: the litmus is "taps in a `std::pmr::vector` read from a **framework** kernel", and a hatch
reads its own `DeviceBuffer`, testing none of that. Proposed signature, one work item per **output** sample:

    [[nodiscard]] constexpr TOut processAt(std::size_t i, std::span<const TIn>... in) const noexcept;

Chunk-relative `i`, which is correct for `i % N` because resampling only ever hands whole frames. Absolute
position (a phase-continuous mixer, spike B) is a later one-line capture of `streamIndex` — not today. Misuse is
bounded by construction: the body cannot write anything but its return value, so there is no race to invite.

**A2 — the hatch form of A is the SECOND row of the table, not the headline.** A monolithic `FastConvolution` on
`processBulk_sycl` batches frames internally, so it escapes the one-frame-per-dispatch limit and will be the
_fastest_ form — §80's "every combination measurable against the others" wants it. But it exercises no interior
device edge and no seated pmr member, so it cannot be the litmus.

**A3 — the taps-as-second-stream idea is DEAD.** A source cannot be auto-parallel (`nInputs > 0` required,
`ExecutionStrategy.hpp:145`, refused explicitly at `:675`), so a cyclic taps stream needs a _serial_ span-tier
block feeding a parallel multiply — slower than the thing it feeds. And it would read a full-rate second stream
where an N-entry table belongs, making the bandwidth-bound multiply ~1.5x slower by design.

**A4 — overlap-SAVE, not overlap-add.** Overlap-add is a scatter-add across frames and is not expressible
race-free in any return-value tier.

**A5 — one addition retires the CAPABILITY hole for A, B, E, D and C; it does NOT give fast reductions.** Every
D/C shape in a real graph is a _segmented_ reduction (RMS per frame, peak per frame, AGC per block) = one work
item per output over its own frame = the indexed tier with `Resampling<N,1>`. A tiled FIR or a tree reduction
needs work-group cooperation, i.e. `nd_range`; `DeviceContextSycl::parallelFor` is a plain `range<1>` that
`.wait()`s after every submit. That is a **second, later tier**, and the naive indexed FIR will be
bandwidth-bound — **predicted here before measuring: the GPU's crossover K will come out LOWER than the SIMD
host's.**

### 💡 MAINTAINER REFRAMING (2026-09-04): a block that CONSUMES LESS THAN IT READS owns its own overlap

_"If the `processBulk(...)` pattern fits the FFT block better, we may also use that instead of `processOne(..)`,
which may eliminate the `HistoryBuffer<T>` definition."_

**REFINED by the maintainer moments later, and the refinement inverts the conclusion:**

_"Using `Stride<>` is still beneficial even if `processBulk(..)` can consume/produce less samples than slots
presented, because this could (potentially) eliminate redundant/confusing code w.r.t. consume/produce etc, but
have the Block<T> present only the samples to the algorithm it actually needs."_

So the two are **not competing answers to one problem** — they are a declarative and an imperative form of the
same thing, and **the declarative one is the better user-level pattern**, which is exactly what these spikes
exist to demonstrate. With `Stride<>` the author writes "window N, hop S" once and the framework presents the
window and owns the accounting; hand-rolled `consume(S)` puts that bookkeeping into every block that needs
overlap, where it can be got wrong silently and where a reader cannot see the intent.

**Therefore the FFT+stride failure is a defect to FIX, not to route around.** The consume-less capability stays
as the imperative escape hatch (and the FIR-needs-no-HistoryBuffer observation still holds), but the headline
pattern for the examples is `Stride<>`.

The consequence of the original observation, still true as far as it goes: **the stride blocker below is not a defect but the wrong mechanism.** A
`processBulk` block already owns its accounting — it can read a window of N and `consume(S)`, and the N-S
samples it did not consume are still in front of it on the next call. Then:

- **FFT overlap-save needs no `Stride<>` at all**: read N, consume S. `d544a0b8` reached for a framework
  mechanism that only serves framework-managed (`processOne`) blocks.
- **a direct FIR needs no `HistoryBuffer<T>`**: read N+K-1, consume N — **the ring IS the history.** The advisor
  independently reached the same shape ("No HistoryBuffer -- the ring is the history").
- **the launch-bound warning may evaporate with it.** `Block.hpp:1705` forces `nResamplingChunks = 1` only when a
  _stride_ is active. Without one, a block can take many frames per dispatch — which is exactly what the GPU
  needs to clear the >=256k-samples-per-dispatch crossover the measurement found.

**Being tested, not assumed** (`qa_Block.cpp`, "a processBulk that consumes less than it reads keeps the
remainder"): a window of 4 hopping by 2 over a ramp must see windows starting 0, 2, 4, ... Must also be
confirmed **on the device span tier**, where `DeviceInputSpan::consume` records into `_acct` and the counts are
replayed onto the real spans afterwards — that replay is the part most likely to differ from the host.

**Not to be over-applied: `HistoryBuffer` stays.** R0 shipped fixed-capacity `HistoryBuffer<T, N>` as _proven_
device state (`DeviceMovingAverage`, `qa_DeviceSpans.cpp`), and consuming-less only replaces it where the needed
history is a contiguous prefix of the input the block is already reading. State that is _derived_ rather than
raw — a running total, an IIR's feedback path, a peak-hold — has no representation in the input ring and still
needs a `mutable` member.

> **PROCESS RULE (learned the hard way, 2026-09-04): never run two builds in the same build directory at once.**
> Doing so produced 21 bogus `null character ignored` errors in `meta/RangesHelper.hpp` — a file nothing had
> touched, and which `git` confirmed unmodified with 0 NUL bytes once the jobs finished. Concurrent `make` in one
> dir corrupts intermediate state and reports it as source corruption. Wait for a background build to complete
> before starting another that shares a directory.

### 🔬 F1 INVALIDATED A BENCHMARK THAT DEPENDED ON THE BUG (2026-09-04)

`qa_FFTPerformance`'s A/B — "what keeping the edge between two device blocks off the host is worth" — built its
_crossing_ arm by putting one FFT on `gpu:sycl` and the next on **`gpu:sycl:0`**: the same GPU, spelled two ways.
Before F1 that produced a host seam, so the benchmark's headline "speed-up from keeping the edge on the device"
was **measuring the string-comparison defect**, not a design choice. After F1 the two spellings are one domain,
the crossing arm stops crossing, and the test fails — correctly.

**Fixed by making the crossing arm cross for real** (`host:sycl` for the second FFT), which restores the test but
changes what it measures: compute placement now moves with the edge, so the two arms are no longer a clean
isolation of edge residency. **They cannot be, and that is a consequence of F1 being right:** two blocks on one
device always share one domain, so no edge between them can be made to cross. Said so in the test rather than
leaving a misleading A/B. Honest numbers on this machine: **24.9 MS/s crossing vs 27.7 MS/s interior** (N=4096).

**This is the strongest evidence F1 was a real defect** — a benchmark had quietly been built on it. It was only
caught because `qa_FFTPerformance` lives in `blocks/*/test/`, which `build-acpp` and `build-gcc15-debug` do not
compile; it needs `build-acpp-registry`. Third config, third finding it alone produced.

### ✅ THE STRIDE DEFECT IS FIXED (2026-09-04) — overlap-save works, and the maintainer's framing was right

`FFT(fft_size=16, stride=4)` over 64 samples now emits **208** samples. The skipped specification test is
un-skipped and green.

**The mechanism, precisely.** `consumeReaders` (`Block.hpp:1109-1129`) acts **only on spans where
`!in.isConsumeRequested()`**. The FFT had called `inSpan.consume(N)`, so its span was skipped and the
`ReaderSpan` destructor performed the block's own request instead (`CircularBuffer.hpp:674-680`); `consume()`
asserts on a second call and there was no way to withdraw a request. So the stride was computed at `:2189` and
then discarded. The fix is two small pieces: `ReaderSpan::releaseConsumeRequest()` (the "framework-owned advance
replaces the block's request" primitive), and `finaliseIO` calling it on each sync span before applying the
stride advance. **Tags follow for free** — `InputSpan::~InputSpan` consumes tags up to the requested count
(`Port.hpp:626-628`), which is now the hop, so tags inside the overlap remain for the next window, matching
`processOne`+stride semantics exactly.

**The device tiers follow with no device-side change**, because every device path returns through the same
`finaliseIO`: the span tier records into `_acct` and replays onto the real span (`ExecutionStrategy.hpp:543-549`)
before reporting `blockManagedIO`, and the sycl hatch does the same. Confirmed by reading, still to be pinned by
a device test.

**`FFT` is the only `Stride<>` user in the tree**, so there was no working `processBulk`+stride block to keep
compatible; the `qa_Block.cpp` stride table is `processOne`-only and is untouched.

#### THE ONE FINDING THE MAINTAINER NEEDS BEFORE SPIKE A IS BUILT

**With the clean `Stride<>` pattern as it stands, an overlap-save chain hands ONE frame per dispatch**
(`Block.hpp:1705-1708` sets `nResamplingChunks = 1` whenever a stride is active). The GPU needs **>=256k samples
per dispatch** to overtake the host (measured). So:

- spike A's frequency-domain side can show a GPU win **only for `fft_size` in 2^18..2^20** (the cap is 2^20,
  `fft.hpp:67`) — a legitimate long-kernel regime, but not at a typical N = 4096;
- **the direct-FIR side is NOT bound by this**: `Resampling<N+K-1, N>` + `Stride<>` lets N be 256k whatever K is.

**The relaxation exists and is small but changes a meaning.** `nResamplingChunks = 1` is intrinsic only for
`processOne` (its loop cannot gather overlapping windows). For `processBulk` the framework could present **k
windows** in one span — `resampledIn = (k-1)*S + N`, advance `k*S` — derivable in `finaliseIO` with no new state,
gated on `HasProcessBulkFunction`. **That changes what a `processBulk` + `Stride<>` span means**, so it is a
separate decision and a separate commit, not folded into the fix above.

#### Hazards to design around in spike A (advisor, verified reading)

1. **EOS drops up to one window.** `isEosPresent = eosAfterSkip < input_chunk_size` (`:2152`) makes the block
   DONE with a partial window unprocessed. Right for overlap-save (the tail is already-processed history) but the
   direct FIR loses up to N+K-2 samples of output at end of stream — visible in every finite `n_samples_max`
   test at N = 256k. `processEpilogue` (`:2237-2249`) is the flush hook; decide "valid-mode, documented" vs
   padding.
2. **Silent starvation above the default ring.** `defaultMinBufferSize` is **65536** (`Graph.hpp:102`) and
   nothing checks `input_chunk_size <= ring size`: a window >= 64k never becomes available and the graph hangs on
   `INSUFFICIENT_INPUT_ITEMS` forever. Spike A must pass `minBufferSize` on every edge (~2x the window). A
   start-time check would turn the hang into an error.
3. **Tags inside the overlap** are honoured only if a later advance lands them on index 0 (`tagWindow = 1`,
   `:1190`, `:1339`); otherwise consumed in a hop silently. Pre-existing for `processOne`+stride and untested.
4. **`sample_rate` substitution** keys on `input_chunk_size != output_chunk_size` (`:1210`), but with a stride
   the real ratio is `output_chunk_size / stride` — an overlap-save FFT's output rate tag is wrong.
5. **S > N (skip) worked by accident** before this fix and must keep working: the skip happens before the next
   chunk via `strideCounter`. Add a `processBulk` row to the stride table.

**Also confirmed:** consume-less-than-read is correct on host and device and is already used in production
(`Trigger.hpp:135`, `StreamToDataSet.hpp:252`, `SyncBlock.hpp:207-215`) — it stays as the imperative escape
hatch. And **my probe for it would have hung at EOS** (no `Resampling`/`min_samples`, so the block is called
forever with a partial window and never goes DONE); removed rather than committed.

### 🛑 SPIKE A IS BLOCKED ON A VERIFIED DEFECT: the FFT's stride does not overlap (2026-09-04)

**`d544a0b8` ("let the FFT be strided, as overlap-save needs") does not work, and nothing tested it.** Measured:
`FFT(fft_size=16, stride=4)` over 64 samples emits **64** samples, where overlap-save needs **208** (13 frames of
16). The setting is applied -- probed `stride=4, input_chunk=16, output_chunk=16` -- and the framework's stride
consume at `Block.hpp:2189` is reached.

**Root cause, verified: the FFT's `processBulk` calls `inSpan.consume(total)`** (`fft.hpp:112`, `:145`), consuming
the whole frame it transformed. A block that manages its own IO overrides the framework's stride consume, so the
overlap never happens. The stride tests that DO pass (`qa_Block.cpp:803-822`, including a real overlap case:
chunk 100 / stride 50 -> 19 calls reading 1900 from 1000 samples) all use **`processOne`**, where the framework
owns the accounting. **Fixing this is a contract change for a self-consuming block, not an oversight** -- hence a
decision, not something to patch quietly.

The overlap test is committed as a **skipped specification** with the root cause in the comment, so it becomes
the acceptance test the moment the contract is decided.

**AND A CORRECTION TO MY OWN EARLIER WORK.** `qa_FFT.cpp` is **not compiled in `build-acpp` or
`build-gcc15-debug`** -- the test dir is gated on `TARGET gnuradio4::GrFourierBlocksShared`, which needs
`GR_ENABLE_BLOCK_REGISTRY=ON`. So when I reported the FFT fence of `4d555e48` as "proven at compile time", the
build had reported zero errors **because it never compiled the file**. Built properly in `build-acpp-registry`,
two things came out:

1. **the fence in `4d555e48` was ineffective.** It required `hostSpan.first(0UZ)`, but `DeviceInputSpan` _has_
   `first()` (`DeviceSpans.hpp:60`), and `subspan()` and `consume()` too. It excluded nothing.
2. **what actually keeps FFT off the kernel tiers is the D7 const flip** (`279d7643`, two commits later): every
   kernel tier requires a `const` body and FFT's is non-const because it drives a stateful host engine. The
   decorative `requires` clause is removed and the real reason is now asserted in `qa_FFT.cpp` rather than
   claimed in a comment.

**Process rule added:** `build-acpp` and `build-gcc15-debug` do NOT build the block-library tests. Anything under
`blocks/*/test/` must be verified in **`build-acpp-registry`** (or `build-ci-clang20-debug`). A green run in the
usual two dirs says nothing about those files.

### 🔧 OPEN FOLLOW-UPS from step 1 (recorded 2026-09-03, none blocking step 2)

**F1 — ✅ CLOSED 2026-09-03** (mutation-verified: 0 interior edges without the fix, 1 with it). DECISION TAKEN (not pre-agreed; recording it here): resolve through a hook registered into
`ComputeRegistry`, exactly as the USM provider already is.** Graph.hpp must stay backend-free, so it cannot call
`DeviceContextRegistry`. Three options were considered: (a) a resolver function registered into the existing
`ComputeRegistry` alongside `ProviderFn`; (b) a new `BlockModel` virtual returning the resolved name, filled at
start; (c) canonicalising `compute_domain` when it is applied. **(c) does not work** — `gpu:sycl` and
`gpu:sycl:0` have _different_ canonical names, so canonicalisation alone never makes them compare equal; only
following the registry's alias does. **(b) is an ABI-append to `BlockModel` and needs a fill pass ordered before
`connectPendingEdges`.** **(a) reuses an established pattern in this exact header** (`register_provider` /
`ProviderFn`), costs one function pointer, and degrades to identity when no device layer is linked — which is
precisely the behaviour a host-only build wants. Taken: (a). The resolver is installed by `registerSyclRuntime()`
next to the USM provider it already registers, so one call arms both.

**F1 — the defect (unchanged):** Step 1 made
the _registry_ agree that `gpu:sycl` and `gpu:sycl:0` are one device, and `qa_DeviceContext` pins that. But edge
placement never calls the resolver: `blockComputeDomain()` reads each block's declared `compute_domain` and
compares verbatim, so two blocks on one GPU spelled differently still get `Access::HostOnly` (a pinned host ring)
and `refuseTwoDeviceDomains` still counts two domains. The fix is to compare `registry.resolve(...).resolved`,
which needs Graph.hpp to reach the device registry — it currently cannot (backend-free by design). Options: a
resolved-name cache filled at start, or hoisting the resolution into `BlockModel`. **This is the one place where
step 1's stated goal is only half-delivered.**

**F2 — RE-ANALYSED 2026-09-03 after F1: the divergence risk is GONE; what remains is a diagnostics
inconsistency.** Both sides now fall back to the _same_ device by construction. `claimUnindexedSpelling` sets
`usm[{kind, -1}]` and the `kind:sycl` alias from the **same** enumeration index in one place, so the execution
ladder's un-indexed rung and the provider's `{kind, -1}` rung name one device. Walking `gpu:sycl:3` on a
two-GPU machine: execution resolves rung 0 (miss) then `gpu:sycl` -> alias -> `gpu:sycl:0`; memory looks up
`{gpu, 3}` (miss) then `{gpu, -1}` -> the same device 0. They agree. `gpu:sycl:1` on that machine resolves
exactly on both sides. **What is left:** execution _warns_ about the downgrade and memory does not, so a graph
placed on a device the user did not ask for says so once rather than twice. **What would break the agreement:**
publishing `usm[{kind,-1}]` and the alias from different devices — they are set together for exactly this
reason, and anything that separates them re-opens the split. Downgraded from a correctness item to a
consistency note; not worth the circuitous change (the provider takes a parsed `ComputeDomain`, so routing it
through the string resolver means building a name, resolving it, and parsing the index back out).

**F2 (original) — `defaultSyclUsmProvider` strips the device index silently** (`SyclRuntime.hpp:97-131`): `{kind, 3}`
misses, falls to `{kind, -1}`, returns the default device's resource with no warning. Execution now resolves
`gpu:sycl:3` loudly to `gpu:sycl`, but edge memory resolves it silently — so the two agree today only because
both land on the canonical device. Make the provider use the same resolution, or they will diverge the moment a
second GPU exists.

**F3 — the downgrade warning is per block, not per graph.** A 200-block GPU graph on a CPU machine emits 200
identical lines. Per-block names the offending block, which is useful; consolidating needs a scheduler-level
pre-pass or shared per-run state. The existing per-type `static` flag (`ExecutionStrategy.hpp:62-66`) is
unsuitable — two graphs in one process share it. **Needs a maintainer decision, not just code.**

**F4 — `DeviceContext::served()` and `DeviceContextRegistry::withdraw()` are now dead.** Step 1 removed their
only consumer (the withdrawn-domain dispatch arm). Nothing in the tree calls `withdraw()`. Delete both, or give
them a caller; leaving an untested mechanism that claims to handle device withdrawal is worse than not having it.

**F5 — `runAndWait()` reports success even when the run ended in ERROR.** Verified: a refused block drives both
itself and the scheduler to ERROR, yet `runAndWait()` returns a value. Every test therefore has to assert on
`sched.state()` separately, and a caller who checks only the return value sees a clean run. Framework-level wart,
pre-existing, worth its own fix.

**F7 — the unreflected-state warning now fires on the sanctioned pattern.** Before D3 an unreflected member was
suspicious; after it, that member IS the mechanism for device-private state. The warning fires once per block
type that does not declare `DeviceStateIsReflected`, which now includes every block that correctly keeps kernel
state. It has been reworded to state the rule (trivially copyable, owns no host storage) rather than to accuse,
but the noise remains and a block that legitimately keeps unreflected state has no way to say so. C++23 cannot
enumerate the omitted members, so the check cannot be made precise. **Maintainer decision: keep as an
informational note, add an opt-out declaration (`DeviceStateIsPrivate`?), or drop it.** A warning that fires on
correct code trains people to ignore warnings.

**F6 — pre-existing, not ours:** `qa_SoapyIntegration` times out on gcc15-debug (LimeSDR hardware test, hangs
standalone, zero references to anything this branch touches; matches the known SoapySource UAF race). The
AdaptiveCpp `CUDA:4` teardown errors are the G10 gotcha — Boost.UT runs suites at static destruction, after
acpp's kernel cache is gone.

### Definition of done

1. A **documented and tested guarantee** that an unreflected trivially-copyable member of a span-tier block is
   device-private persistent state: excluded from the settings surface, seated once, not re-seated from the host
   on a settings change, and not requiring `copyBackUserState`. (Was "a declared category" -- the declared type
   was withdrawn 2026-09-03; see D16.)
2. Tag/message forwarding for a same-domain chain that does not round-trip through the host (after establishing
   whether it currently does).
3. Tests: state retained across dispatches; survives a settings-epoch bump; never appears as a setting; a
   same-domain chain forwards without a host hop. Each mutation-tested -- a test that passes without the fix is
   worse than none (§80.18 was nearly shipped that way).
4. ✅ **CLOSED 2026-09-03.** `HistoryBuffer` usable as device state -- for the **fixed-capacity** form
   (`HistoryBuffer<T, N>`), which is `std::array`-backed and therefore trivially copyable and self-contained.
   The **dynamic-extent** form remains ineligible and always will be: it owns a `std::vector` whose pointer a
   kernel would follow back to host memory. Pinned by `DeviceMovingAverage` in `qa_DeviceSpans.cpp`.

## ⚡ READ FIRST — current state & how to resume (2026-08-28)

> **AGENT BUDGET — §0 rule 5 is binding.** One Opus/Fable agent at a time; at most 4-5 Sonnet agents, and only for
> mechanical refactor/test work. Design and analysis stay with the orchestrator.

> **STANDING STYLE RULES — §0.6 and §0.7, user directive 2026-08-25.** _Nomen est omen_: name methods, lambdas,
> parameters and variables for their functional use instead of explaining them in a comment; reasoning goes in the
> commit message or this file, never in the source. And extract to an **in-method lambda** by default — promote to a
> free-standing function only when it is generic AND unit-testable AND has callers outside the method that spawned
> it. Read §0.6/§0.7 before writing code here. (Also durable, since this file is temporary:
> `feedback_nomen_est_omen_over_lambdas` in Claude's memory.)

> **`/tmp` IS A 16 GB tmpfs ON THIS MACHINE.** When it fills, every tool that captures output starts failing in
> confusing ways -- shell commands return errors with no output, redirects create empty files, git returns 128,
> and the harness reports exit codes that are not the command's. The tell is `EDQUOT` from a plain file write.
> `/home` has ~900 GB free, so build trees are never the cause; the consumers are Claude session scratch dirs
> under `/tmp/claude-1000/`, clangd `preamble-*.pch` (~100 MB each) and AdaptiveCpp's JIT cache. Point `TMPDIR`
> at `/home` for build-heavy sessions. Cost half a session on 2026-08-30 before the errno gave it away.

> **RULE EARNED THE HARD WAY, 2026-08-21 and again 2026-08-25: re-verify any OPEN item against the code before
> working on it or quoting it.** Every ToDo list in this file below the line has carried items the code had already
> closed. The lists in §13, §14, §18.2 and §28 date from July/August and are records, not instructions.

> **GOLDEN REFERENCE (maintainer, 2026-08-30): the first 20 commits of `syclExperiments` are FROZEN.** Do not
> amend, rebase or reword them. The DSP spikes (§80) are **independent commits stacked on top of this branch**, so
> that a bug or gap found while building them can be promoted down into the frozen part as its own fix.

**Branch `syclExperiments` — 20 commits above `origin/main`, 0 behind, and PUSHED**: `origin/syclExperiments`
matches `HEAD` exactly as of 2026-08-30 (leased force-push after the N1 squash). Any further history rewrite needs
another leased force-push; ask first.
Tree clean. **Build is clean end to end** (full AdaptiveCpp build, 0 errors, both registry ON and OFF) and
**`ctest` is 100/100** as of 2026-08-27. The three former failures were each a test failing on something it does
not test — a build option it never checked for (`qa_SubGraphAssets`, `qa_Graph`) and the argmax of a monotonic
step (`qa_DataSetEstimators`) — fixed in one commit at position 4, disjunct from the feature work. There is no
§78; that pointer was always dangling.
`pr_message.md` is the PR draft and follows the maintainer's own PR style (see #820).
**This file and `pr_message.md` are the only untracked scratch docs** — `featDeviceIntegration_findings.md`,
`featDiagnosticChannel_design.md`, `sycl_format.md` and `webgpu_mvp/` have been moved out of the repo, so any
reference to them below is dangling.
Safety refs: the branch is pushed and `origin/syclExperiments` matches `HEAD` exactly (0 ahead, 0 behind), so the
remote is the backup. All `backup/*` branches were deleted as spent (the last on 2026-08-30, after the push). What still matters is kept deliberately: the tag
`snapshot/webgpu-glsl-complete-20260824` (the branch with the shader backends still in core, the state D5 removed)
and `spike/p2300-execution` (the P2300 work extracted out of this branch).

### 📐 §80 · DSP SPIKES — the plan (maintainer, 2026-08-30)

> **BLOCKED BY R0** (top of this file): spikes A2, B and the filter migration all need device-private block
> state, which the framework has no category for. Close R0 first.

**Purpose.** Exercise the device design with real DSP rather than synthetic blocks. The FFT is the only real
workload it has seen so far.

**Priority order, and it is not negotiable:**

1. **functional and numerical correctness** — first, and on its own terms
2. **benchmarkable** — every combination (type x domain x form) measurable against the others
3. **performance / industrial integration** — third, not a gate on the first two

**Structure.** Split into **algorithms** (`algorithm/include/gnuradio-4.0/algorithm/...`) and **blocks**
(`blocks/...`), matching what the repo already does. **Each feature is an independent commit** stacked on the
frozen 20.

**Types: all four.** `float`, `double`, `gr::complex<float>`, `gr::complex<double>`.

**Work quantum:** each block declares its own minimum required input samples, for both the `processOne` and the
`processBulk` form. This is how a spike escapes the batch = 1 regime where §53 says the GPU always loses.

| #      | spike                                                            | what it tests that nothing else does                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| ------ | ---------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **A**  | **FIR + IIR filter** (one feature, both forms)                   | fast convolution `FFT -> multiply -> iFFT` is §68.1's litmus: the two-input multiply (N-ary), interior device-only edges, complex end to end, and taps in a `std::pmr::vector` — the exact shape whose canary hole was found and fixed. The **premise to measure is the crossover**: kernel length at which frequency domain overtakes direct convolution. It moves on GPU, because the FFT itself only wins at N >= 4096.                                                                                              |
| **A2** | **IIR — kept inside A deliberately**                             | maintainer's reasons: (a) keeps data on the GPU; (b) far fewer multiply/adds than a functionally equivalent FIR; (c) an educational case to benchmark against; (d) batches over N samples; (e) pays most with **many channels in parallel**, on GPU and on CPU via SIMD. Tests the **span tier** — one work item, stateful, residency rather than parallelism — which no real block has exercised.                                                                                                                      |
| **B**  | **Channelizer** (GR3's frequency-xlating filter; polyphase bank) | single-in, many-out: N-ary **outputs** with real work, fan-out over one shared device ring, and polyphase decimation exercising the span tier's **independent in/out counts** (the feature whose absence hung the graph). §68.1 names it the workload where interior traffic exceeds boundary traffic by the channel count. **Caution (§53):** 64-1024 channels sit _below_ the FFT crossover on transform size, so it must be justified by transfers removed, not compute — decide what to instrument before building. |
| **C**  | **Correlator**                                                   | FFT-based correlation overlaps A, but the **peak search is a reduction**, and no tier serves reductions: the auto-parallel tier is strictly element-wise. Expected to surface a design gap rather than confirm one.                                                                                                                                                                                                                                                                                                     |
| **D**  | **Reduction / statistics** (RMS, power, peak-hold, AGC)          | the same gap stated directly. Ubiquitous in real graphs. If every reduction must drop to `processBulk_sycl`, that is worth knowing now.                                                                                                                                                                                                                                                                                                                                                                                 |
| **E**  | **Rational resampler**                                           | the span tier's separate in/out counts are new and covered only by a synthetic upsampler. Cheap, honest coverage.                                                                                                                                                                                                                                                                                                                                                                                                       |

**Suggested order:** A (litmus, unblocks the crossover number) -> B (strongest design test) -> E -> C/D last.

#### 80.0 Decisions taken (maintainer, 2026-08-30)

- **Transfer/metrics counter: REINSTATE, but optional.** Compile-gated, pay only for what you need. It is the
  instrument §68.4 requires before spike B's claim ("interior traffic removed") can be measured at all.
- **Benchmarks default to `float`**; `double` is exercised for basic numerics and compatibility, not for
  performance headlines.
- **IIR batching means BOTH:** (i) across channels — the obvious parallel win, on GPU and via CPU SIMD; and
  (ii) **within one stream, over a batch of samples** — process 8 / 16k / 64k in a single call instead of
  invoking the core function per sample and handing one sample onward at a time. (ii) is about amortising
  invocation overhead, not about breaking the recursion, which stays sequential.
- **Extend the existing filters where possible** rather than adding parallel implementations.
- **Numerical oracle:** compare against the time-domain computation using the existing algorithm, which is
  thoroughly tested. No external golden vectors needed.

**Constraint the counter must respect — this branch already fought it.** Commit _"give Block one layout whether
or not a device backend is compiled in"_ made `Block<T>` carry its device state unconditionally (~40 B) precisely
so `sizeof(Block<T>)` cannot differ between translation units linked into one binary. A conditionally-compiled
counter member in `Block<T>` would reintroduce that ODR hazard. Put the counters in `DeviceContext`, which is
already behind the backend gate, or make them unconditional-size like the shadow.

**Falls out of the design, worth stating:** an IIR **cannot** use the per-sample tier on a device — its state is
written by the body, so the self-mutation probe refuses it. It must use the span tier. The `processOne`-vs-
`processBulk` comparison the maintainer wants for the educational case (§80 reason (c)) is therefore the same
comparison the tiers already force.

#### 80.3 PRE-COMMIT: migrate STL containers in common blocks to pmr (maintainer, 2026-08-30)

Ordered **before** the spikes, as its own commit, with the unit tests extended in that same commit, because every
later block builds on it.

**The rule the audit must apply — both halves, verified 2026-08-30:**

1. the member must be a **pmr container** (`std::pmr::vector<T>` etc.) — `PmrMigratable` is what makes it
   `DeviceSeatableContainer`, and
2. it must be **listed in `GR_MAKE_REFLECTABLE`** — `Block`'s migration loop walks `refl::data_member<kIdx>`
   (`Block.hpp:1503`) and the relocatable gate folds over the same set, so an unreflected pmr member is neither
   seated nor seen (§80.2).

**`std::string` settings are a hard disqualifier for the framework tiers.** `DeviceSeatableContainer` excludes
`basic_string` by design (`DeviceRelocatable.hpp:40` — SSO data lives inside the object), and `std::pmr::string`
does not help. **15 blocks** currently carry an `Annotated<std::string…>` setting, including `fourier/fft.hpp`
and `filter/SavitzkyGolayFilter.hpp`. Such a block can still reach a device, but **only through
`processBulk_sycl`**, which is the one arm of `canDispatch()` that does not require `DeviceRelocatable` — the
block stays on the host and only its data travels. That is exactly how the shipped FFT block works.

**`BasicFilterProto` is clean on that count** — its reflected settings are enums, floats and `Size_t`, no string —
so it is a viable candidate for the framework tiers once `_filter` is resolved.

**Scope warning:** the migration cascades. `Section<T,bufferSize>` holds `HistoryBuffer<T,bufferSize>` twice plus
a plain `std::vector<T> autoCorrelation`; `HistoryBuffer` must become allocator-aware too, or the filter keeps a
member that cannot be seated.

#### 80.4 Why reflection is required on the device, and the `std::string` audit (2026-08-30)

**Mechanism (asked by the maintainer).** The block object reaches the device by `relocateBlockToDevice`, which is
a plain `memcpy` of `sizeof(TBlock)` — **every member's bytes travel, reflected or not**. Reflection is not what
moves them. It drives everything _about_ them:

1. `migrateFieldsToDeviceResource()` -> `rebindFieldsTo(mr)` (`Block.hpp:1001`, `:1496`) walks
   `refl::for_each_data_member_index` and re-seats each pmr container's **storage** onto the device resource. An
   unreflected pmr member keeps host storage: its control block is copied, and the kernel follows a host pointer.
2. `DeviceRelocatable` folds over the same set to decide eligibility — hence §80.2.
3. `copyBackUserState` returns reflected trivially-copyable members after a single-work-item kernel.
4. `firstStaleMirrorMember` is the debug coherence check.

So: _bytes travel regardless; reflection decides whether the block may go, whether its heap data goes with it, and
what comes back._

**The `std::string`-as-enum hypothesis is CONFIRMED, with a smoking gun in `fourier/fft.hpp`:**

```
Annotated<std::string, "window", …> window = enumName(window::Type::Hann);   // :180  stored as text
gr::algorithm::window::Type         _windowType = window::Type::Hann;        // :196  shadow enum
_windowType = parseEnum<window::Type>(window).value_or(_windowType);         // :310  parsed back
```

Pure indirection — and `BasicFilterProto` already uses `Annotated<algorithm::window::Type, "fir_design_method">`
directly, so the enum is first-class today. Replacing it deletes the string, the shadow and the parse.

**Classification of the in-scope strings:**

- **enum stand-ins, removable:** `fft.hpp` `"window"`; `SavitzkyGolayFilter` `"alignment"` (Centred/Causal) and
  `"boundary policy"` (Reflect/Replicate).
- **genuine free text, not enums:** `"signal name"`, `"signal unit"`, `"signal quantity"`, `"DataSet name"`
  (SigMF / ISO 80000-1 metadata) and `"trigger name"` / `"signal_trigger"` / `"context name"` / `"filter"`
  (arbitrary tag names). These sit on sinks and sources — `DataSink`, `FunctionGenerator`, `SyncBlock` — not on
  the compute blocks the spikes need on-device, so the compute path can plausibly become string-free.

#### 80.5 Pre-commit scope, verified against the code (2026-08-30)

**`Annotated<enum>` is first-class — re-checked, and the wire format makes the swap SAFE.** 12+ existing uses
across 8 files (`filter::Type`, `filter::iir::Design`, `algorithm::window::Type`, `FilterType`,
`function_generator::SignalType`, `fileio::CompressionMode`, …). `Settings.hpp` supports it directly:
`std::is_enum_v` is in the supported-setting predicate (`:40`), `parseEnum<T>(str)` converts an incoming **string**
into the enum (`:197`), and `enumName()` writes the **name** back out (`:209`). So replacing
`Annotated<std::string,"window">` with `Annotated<window::Type,"window">` is **wire-compatible**: a saved graph or
YAML carrying `window: "Hann"` keeps working. The compatibility risk raised earlier does not exist.

**`HistoryBuffer<T, N, Allocator>` — the maintainer's assumption is correct.**
`buffer_type = conditional_t<N == dynamic_extent, std::vector<T, Allocator>, std::array<T, N*2>>` (`:70`).

- **static N** -> `std::array<T, N*2>` stored **inside** the object: trivially copyable when `T` is, relocatable
  with no work. Leave it alone, as instructed.
- **dynamic N** -> `std::vector<T, Allocator>`, and the **`Allocator` template parameter already exists**
  (`:67`, defaulting to `std::allocator<T>`). Making it pmr-aware is therefore additive, not a redesign:
  add `using allocator_type = Allocator;`, an allocator-extended constructor, and the
  **move-plus-allocator** constructor that `PmrMigratable` actually tests for
  (`is_constructible_v<T, T&&, polymorphic_allocator<>>`). Currently only `HistoryBuffer()` and
  `HistoryBuffer(std::size_t capacity)` exist (`:118`, `:120`).

**CORRECTED 2026-08-30 (adversarial review, verified).** An earlier draft here claimed the _dynamic pmr_ form
was the one that could run on the per-sample tier and the static form could not. **That was wrong.** Only
`_buffer` and `_mirrorDirtyCount` are `mutable` (`HistoryBuffer.hpp:72,77`); `_write_position` and `_size` are
**not**. Every state-advancing operation writes them, so a stateful filter's `processOne` cannot be `const` in
_either_ buffer form, and never reaches the per-sample tier at all — which is what §80.0 already said. There is no
static/dynamic asymmetry at the decision point. **The rule is simply: stateful ⇒ span tier or `processBulk_sycl`,
both forms.** Motivate the pmr work by the span tier and heap seating, not by per-sample eligibility.

**`HistoryBuffer` is used widely across the codebase** — its own unit tests must stay green, and the pre-commit
extends them rather than only adding new ones.

#### 80.2 GAP FOUND IN THE FROZEN PART (2026-08-30) — the relocatable gate only sees _reflected_ members

**Verified by compiling a probe, not by reading:** a block whose host-pointer-bearing member is omitted from
`GR_MAKE_REFLECTABLE` passes `DeviceRelocatable`. The same member, when reflected, is correctly refused.

```
static_assert(!DeviceRelocatable<VisibleVector>);  // passes  -- reflected std::vector is refused
static_assert(!DeviceRelocatable<HiddenVector>);   // FAILS   -- unreflected std::vector is admitted
```

Cause: `DeviceRelocatable` folds over `refl::data_member_count`, which counts only members the macro registered.
An unreflected member is invisible to it yet present in `sizeof(TBlock)`, so it is bit-copied to the device and
its pointer followed there. `firstNonRelocatableMember()` also has nothing to name, so the diagnostic is silent.

**This is not hypothetical for §80:** `BasicFilterProto::_filter` is exactly that shape — a `filter::Filter<T>`
holding `Section<T,…>` with `HistoryBuffer`s and a plain `std::vector<T> autoCorrelation`, and it is **not** in
that block's `GR_MAKE_REFLECTABLE`. The block we intend to extend is the case that escapes the gate.

**A complete fix needs C++26 reflection** — C++23 cannot enumerate unregistered members. Options, unranked:

1. **Heuristic**: compare `sizeof(TBlock)` against the extent covered by reflected members plus the base; refuse
   when unexplained bytes remain. Catches `_filter`; padding and alignment make it approximate.
2. **Opt-in assertion** a device-eligible block must carry, making hidden state the author's declared problem.
3. **Document only** — weakest, and inconsistent with a branch whose stated rule is that nothing is refused
   silently and nothing is substituted silently.

Note the mutation canary is unaffected: it `memcmp`s over `sizeof(TBlock)` and so _does_ see unreflected bytes —
but it detects writes, not a host pointer being dereferenced on a device.

**Belongs in the frozen part as its own fix, promoted down** (this is the case §80's stacking rule anticipated).

#### 80.6 ADVERSARIAL REVIEW (fable, 2026-08-30) — findings I verified and accepted

Three checked against the code by me, not taken on trust:

- **`Section` can never be seated, so pmr-ifying `HistoryBuffer` does NOT unblock `_filter`.**
  `DeviceSeatableContainer` requires **trivially-copyable elements** (`DeviceRelocatable.hpp:40-45`).
  `FilterCoefficients` holds `std::vector<T> b, a` (`FilterTool.hpp:88-92`) and `Section` adds
  `std::vector<T> autoCorrelation` — in the static-buffer variant too. So `pmr::vector<Section>` is refused
  whatever we do to `HistoryBuffer`. **Nested pmr does not compose.** §80.3's cascade note aimed at the wrong
  target: gate-passing needs a **trivially-copyable device `Section`** — inline fixed-capacity a/b/history,
  `autoCorrelation` hoisted out — inside one flat pmr container. Also: `HasFilterCoefficients` requires
  `t.a` convertible to `std::vector<value_type>&` (`FilterTool.hpp:95-99`), so migrating `FilterCoefficients`
  silently un-matches every algorithm constrained on it.
- **The per-sample-tier inversion claim was wrong** — see the correction in §80.5.
- **The device FFT is float-only** (`fft.hpp:117`, `requires std::same_as<T, float>`). Of the four planned types
  only `complex<float>` reaches the device tier; `double` warn-once falls back, so any "crossover" measured for
  double is a CPU measurement. Either state the device path is float-only, or put SyclFFT double support in scope.

Accepted without independent re-verification (reviewer cited file:line for each):

- **Overlap-save cannot be expressed today.** The stream FFT is strict 1:1 (`fft.hpp:76-80`) and declares no
  `Stride`. Without overlap, FFT-based convolution computes **circular** convolution — it fails the time-domain
  oracle, i.e. priority 1. Either the FFT block grows stride/overlap, or the chain grows a fourth stage, or
  fast-conv becomes one self-contained block — **in which case there are no interior edges and that credit for
  spike A evaporates**. This is the real work §80.1's costing omitted.
- **The shipped N-ary `Multiply` cannot reach a device tier** — dynamic `std::vector<PortIn<T>>`
  (`Math.hpp:86`), and no tier accepts a dynamic port collection. Spike A needs a **new static two-input complex
  multiply**; drop "N-ary" from A's credit line.
- **Same blindness hits B harder.** A runtime-channel-count channelizer needs `std::vector<PortOut<T>>`, also
  unreachable. Built with M static ports it is not a realistic channelizer; built as a sub-graph of M filters on
  one ring it exercises only fan-out, which §68 already measured. **"Teach the tiers dynamic port collections"
  may be the largest work item hiding in this plan** — name it before B starts.
- **Split-state hazard for stateful span-tier blocks.** `copyBackUserState` returns only trivially-copyable
  non-pmr reflected members. With a top-level pmr HistoryBuffer the kernel advances the delay-line _contents_ in
  shared USM while the ring _indices_ advance only in the mirror; on a settings-epoch bump `deviceMirror`
  re-copies from the host object and clobbers them, silently. Rule to adopt: a span-tier block's mutable state
  lives **either** entirely in seated heap **or** as top-level trivially-copyable reflected scalars — never split.
- **`HistoryBuffer::resize()` is a latent pmr bug** (`:272-296`): it builds `std::vector<T,Allocator>` with a
  **default-constructed** allocator then `std::swap`s. With pmr that is UB on unequal non-propagating allocators.
  Must construct from `_buffer.get_allocator()` and assign.
- **Silent unseat window:** `migrateFieldsToDeviceResource` early-outs when the resource already matches
  (`Block.hpp:1007-1010`), and `staleMirrorDiagnostic` is Debug-only _and_ blind here (after an epoch bump the
  mirror is memcpy'd from the block, so both agree on the bad pointer). Tests must assert **numerics** after a
  mid-stream settings change on a device domain — pointer-identity checks cannot catch it.
- **Spike C is redundant** (= A's chain + D's reduction) and its "discovery" is already a static fact. Fold its
  correlation into A's qa vectors; keep D as one decimating reduction with a `processBulk_sycl` reference to
  force the real decision (add a reduction tier vs bless the hatch). **AGC belongs in A2, not D** — it is a
  recursive per-sample loop, not a reduction.
- **The table's "what it tests that nothing else does" overstates**: pmr taps, mid-run re-seat, 2-in/2-out
  kernels, interior device edges and fan-out are all already covered synthetically. The spikes' defensible value
  is real DSP numerics, multi-block chains, and forcing the reduction / dynamic-port / overlap decisions.

#### 80.8 `qa_FFTDevice` — FIXED (2026-08-31). Four defects, two of them in the product

The suite had **never once run to completion** since it was written: it aborted at teardown, so its assertions
were never evaluated. `2590 asserts in 54 tests` now pass, on the **GPU**.

**Product defects (the ones that matter):**

1. **`DeviceContextSycl` stored a raw pointer to a caller-owned queue.** `FFT` caches such a context from whatever
   queue reaches `processBulk_sycl` and frees USM through it _in its destructor_ — long after a caller's local
   queue has died. Now the context owns the queue (a `sycl::queue` is a refcounted handle). Proven load-bearing:
   reverting it segfaults in `~FFT() -> sycl::free`, reproduced 6/6 with a fresh appdb.
2. **`processBulk_sycl` passed `window_coefficients.data()` — a host pmr vector — into a GPU kernel.** Through a
   graph the settings migration re-seats that member into USM; called directly, nothing does, and this card has no
   HMM (`nvidia-smi -q` -> `Addressing Mode: None`), so the dereference is `cudaErrorIllegalAddress` (CUDA:700).
   The block now stages the window into device scratch when it is not device-accessible. **This is what let the
   test keep its default (GPU) queue** — the earlier `cpu_selector_v` workaround was deleted, so GPU coverage of
   `processBulk_sycl` is retained rather than lost.

**Test defects:** 3. **G10 violated:** 7 namespace-scope `const suite<>` in a kernel-bearing TU. Now driven from `main()`, each
former suite nested under a named test — the names matter: collapsing them into one `global` suite makes
duplicate test names collide and Boost.UT then runs **nothing at all, silently**. 4. **Phase asserted at vanishing-magnitude bins.** Only `phase[5,9,13]` failed while the tone is at bin 6 and
`magnitude`/`re`/`im` all passed: `atan2` of two vanishing components follows the rounding, not the signal.
`expectSpectraMatch` now compares phase only where the magnitude carries it.

**Corrections to claims made while debugging — do not repeat them:**

- "cpuSimdTests-only still aborts" — **false**, drawn across stale binaries (three existed within ~15 minutes).
  No CUDA work happens before the spectrum/unwrap tests at all.
- "fails with `ACPP_VISIBILITY_MASK=omp` too" — **false**, that run exits 0.
- The abort was **not** static destruction: a handler-less `queue{}` dying at its scope brace calls
  `throw_asynchronous()` -> hipSYCL's default handler -> `std::terminate`. G10 is real and its fix stands, but it
  was not this abort's mechanism.
- **The appdb kernel cache is a heisenbug amplifier:** a crashed run never persists it, a clean run does, and a
  rebuilt binary hashes to a cold entry — which is what made the pass/abort/segv observations contradict.

**Standing note:** `qa_PythonBlock` fails in `build-acpp-registry` (Subprocess aborted, inside Python iterations).
It is **pre-existing and not ours** — `git log origin/main..HEAD -- '*PythonBlock*'` is empty. Registry build is
therefore 137/138.

#### 80.9 Split-state hazard: DOCUMENTED, not detected (2026-08-31)

The maintainer chose a warn-once diagnostic. **It cannot be built precisely, so the rule is documented instead**
(`USER_API_GPU_Blocks.md`, "Where a stateful block keeps its state"):

- `copyBackUserState` skips every `PmrMigratable` member (`DeviceRelocatable.hpp:158`), so a seated container's
  object-level fields never return from the device. That is correct for the storage and wrong for bookkeeping
  held _inside_ the container object.
- The dangerous case (`HistoryBuffer`: ring indices in-object, advanced by the kernel) is structurally
  indistinguishable from the benign one (`pmr::vector<T>` taps: only ptr/size/capacity in-object, never written
  by a kernel). A warn would fire on every block with seated taps and carry no signal.
- **No block holds a pmr `HistoryBuffer` today** — the hazard has no instance and becomes reachable exactly when
  the device IIR (spike A2) is built. Revisit the diagnostic then, when there is a real case to key on.

#### 80.10 `std::complex` cannot be multiplied inside a kernel (measured 2026-08-31)

Dispatching `MultiplyPair<std::complex<float>>` to a device fails with **CU:218 (invalid PTX)**: the libstdc++
`operator*` lowers to a libgcc complex helper (`__mulsc3`) that has no device implementation. **The failure is
context-poisoning, not local** — the next test in the same binary, an unrelated `gr::test::Gain`, failed with the
same code until the offending block was removed. This is why `gr::complex<T>` exists, and why `fft.hpp`'s kernels
use `gr::complex<value_type>` throughout.

**Consequences for spike A:** the fast-convolution multiply must be `gr::complex`, not `std::complex`.
The four pair blocks are now registered for **both** complex types — `std::complex` is correct on the host,
`gr::complex` is the one that survives a kernel. `qa_DeviceAutoParallel` covers this: _"the two-port multiply
carries gr::complex through a kernel"_ runs `MultiplyPair<gr::complex<float>>` on the device with zero CPU
fallbacks and checks both the real and imaginary cross terms. The same graph built on `std::complex` is what
produced the 12 CU:218 failures, so the test is a regression guard, not a demonstration.

**Pulling `Complex.hpp` into a widely-included header broke an unrelated benchmark, and the mechanism is worth
keeping.** `Complex.hpp` declares `gr::abs(complex<T>)`. `ImGraph.hpp` opens `namespace gr::graph` and called
unqualified `abs()` on an `int`, relying on `::abs` leaking in from a C header. Once unqualified lookup finds
_any_ `abs` in the enclosing namespace `gr`, **it stops** — `::abs` is never considered — so the call failed to
compile the moment `bm_Scheduler.cpp` included `Math.hpp` ahead of `ImGraph.hpp`. Constraining `gr::abs` would
not have helped: name hiding keys on the _name_, not on whether an overload matches. Fixed by qualifying the
call `std::abs` (+ `<cstdlib>`).

**The other three sites are NOT the same trap — checked 2026-09-01, no fix needed, do not "fix" them.**
`SVD.hpp:47` and `DataSetUtils.hpp:61-62` carry `using std::abs;` in scope, which is the correct two-step ADL
idiom: block-scope lookup finds `std::abs`, and ADL still reaches `gr::abs` for a `gr::complex` argument.
`ImCanvas.hpp:113`'s `abs` is a function-local lambda declared 16 lines above it, so it hides everything and is
immune by construction. Verified by running `gr::math::detail::sign` on both complex types: both normalise to
(0.6, 0.8), which only happens if each spelling resolves to its own correct `abs`. `ImGraph.hpp` was defective
precisely _because_ it lacked the `using` and leaned on `::abs` leaking from a C header.

Registering `gr::complex` required adding `#include <gnuradio-4.0/Complex.hpp>` to `Math.hpp`: the block-lib
generator emits a TU that includes _only_ the registering header, so any type named in `GR_REGISTER_BLOCK` must
be reachable from it. A stale build directory hides this — the generator does not re-run on a header edit alone,
so **a registry check needs a reconfigure, not a rebuild**, or it silently tests the previous registration list.

Two follow-ups worth having, neither done:

- the device seam should refuse a block whose kernel cannot be built, rather than let one poison the context for
  everything after it;
- **`gr::complex` is a usable block type but not a usable _setting_ type.** `TagSource<gr::complex<float>>` is
  rejected with _"annotated member 'values' has unsupported setting type"_. The cause is not the sample converter
  in `TagMonitors.hpp` — it is the wire layer. **Measured, not assumed (2026-09-01):** adding `gr::complex` to
  both `Settings.hpp` predicates _and_ a `SampleValueConverter<gr::complex<T>>` still fails, at
  `ValueMap.hpp:790` — `static_assert(TensorElementType<ElemT>)`, "Tensor element type must be an inline scalar,
  std::complex<double>, or gr::pmt::Value". The blocker is a type-tag chain, not a predicate: **38 sites across
  `ValueMap.hpp` (18), `Value.hpp` (11) and `Value.cpp` (9) enumerate `std::complex`**, and `gr::complex` would
  have to be threaded through all of them, plus a decision on round-trip identity (the two are layout-compatible
  and implicitly convertible both ways, so a shared wire tag is possible but erases which type went in).
  **That is the §12.3 wire/identity decision, not a drive-by fix** — 38 wire sites to delete a 14-line test
  block is a net complication, so the "prefer `gr::complex` where it simplifies" rule points the other way here — until it lands, the device complex test takes a float ramp from `TagSource` and builds its
  samples with a test-local `ToComplex` block, whose only setting is a `float`.

#### 80.11 `gr::complex` is now a settings type, and AdaptiveCpp will not rescue `std::complex` (2026-09-01)

Widened and landed as _"feat(core): accept gr::complex wherever a setting takes std::complex"_, rebased to sit
directly after the wire-map fixes (`88b8d0c7`) because `gr::complex` comes from `8a5fdaff`, which is on main.

The earlier "38 wire sites" figure was **wrong** — it counted mentions, not gaps. Probing each path
individually found only 3 real gaps; construction already worked through the implicit conversion. The two
spellings share one `ComplexFloat32/64` tag rather than gaining a second, so the format is unchanged and a
value stores under either spelling and reads back under either. Comparison is deliberately left to the implicit
conversion: instantiating `operator==` for the second type would make every comparison against a complex
ambiguous between two equally viable candidates.

**Probe-design lesson:** one of the three "gaps" was a bug in the probe — it called `ValueMap::insert(key,
value)`, an overload that exists for no type at all. It only surfaced once a `std::complex` control column was
added. A capability probe without a known-good control cannot tell "unsupported" from "I called it wrong".

**AdaptiveCpp upgrade: NO, and stop asking.** Verified 2026-09-01 against upstream. The installed toolchain is
`25.10.0+git.2e93a5a8`, a local build already **56 commits past the v25.10.0 tag** (the Dockerfile pins
`ACPP_GIT_REF=v25.10.0`, so local and CI have drifted). `StdBuiltinRemapperPass.cpp` — the pass that would carry
such a mapping — is **byte-identical** between v25.10.0 and develop tip, lists real-valued libm only, and none
of the 76 shipped bitcode libraries defines `__mulsc3` or `cabsf`. Issue #341 (a `sycl::complex`) has been open
and unimplemented since 2020 and would not fix `std::complex` anyway. Also corrected: **#340 is not a second
instance of this failure** — it reports a silent miscompile and was closed by its reporter as
environment-specific; it is citable for the maintainer's guidance only. `Complex.hpp` now records all of this.
Predicted but unmeasured: `operator/` (`__divsc3`) and `std::exp`/`log`/`polar` on complex fail identically.

#### 80.12 `StackValueMap` — keep, but it hides a duplicated rule (evaluated 2026-09-01)

Came in with `5c461024` so a kernel could build a tag payload without allocating. It is **not** an
implementation duplicate of `ValueMapView` — it owns no map logic, just a byte array plus a view over it.

It earns its keep on one specific ground: naming the capacity makes the slot constraint checkable at the call
site (`static_assert(decltype(payload)::kCapacity <= kDeviceTagSlotBytes)`), which a raw `std::array` cannot
state without re-deriving the layout rule.

**BOTH FIXED (2026-09-01)**, split by where each rule originates: the shared sizing helper
(`entryCapacityForKeys` / `blobBytesForKeys`) went into the commit that introduced `try_emplace_map`, and
`StackValueMap` picks it up and loses the forwarder in the device commit. `nKeys + 1U` now appears exactly once
in the file.

Two defects, and one is the real find:

- **The sizing rule is encoded twice in one file, identically, comment and all**: `ValueMap.hpp:1502`
  (`try_emplace_map`) and `ValueMap.hpp:3824` (`StackValueMap`) each compute `nKeys + 1` "formatAt writes an end
  marker" followed by `sizeof(Header) + entryCapacity * sizeof(PackedEntry) + payloadBytes`. That is the thing
  that rots silently if the wire layout changes. **This — not `StackValueMap` — is the genuine generic
  `ValueMap` cleanup worth moving to the front of the branch.**
- Partial facade: it forwards exactly one method (`try_emplace`), so one object gets two spellings. Forward all
  or none; none is the KISS call.

**Retracted:** the earlier suggestion to move `StackValueMap` itself to the front. One use site, and it is a
test; no production consumer; purpose is entirely device-tag scaffolding. It stays with the device commit.

#### 80.13 ENVIRONMENT: clang-21 was removed from this machine on 2026-09-01 07:35 — acpp is broken

`clang22-22.1.8-3.1` was installed at **07:35:15** and took clang-21 with it. AdaptiveCpp's
`/usr/local/etc/AdaptiveCpp/acpp-core.json` still names `"default-clang": "/usr/bin/clang++-21"`, which no
longer exists, so **every acpp invocation now fails** — a trivial `int main(){}` included. clang 18/20/22 are
present; 21 is gone entirely, and `/usr/lib64/clang/21/include` is only partly removed, which is why the first
symptom was a bogus `'__stdarg_header_macro.h' file not found` rather than a missing compiler.

**This bisects the day's test results by clock, so check timestamps before trusting one:** `build-acpp` built
and tested at 07:07 (93 real compiles, 100% of 100) and is VALID; the `build-acpp-registry` half of the same
sweep ran 07:37-07:38 and is an environment casualty, not a code failure. Needs root to repair — reinstall
clang-21, or rebuild AdaptiveCpp against clang-22 (the acpp plugin is an LLVM-version-bound artefact, so
repointing the json at clang-22 is not expected to work on its own).

Unrelated but adjacent: a partial `rm -rf build-acpp-registry/plugins/<X>` plus a reconfigure leaves that build
dir inconsistent and produces `undefined reference: gr_blocklib_init_unit_*` for _other_ plugins. A full
reconfigure clears it. Do not read those link errors as a source defect.

#### 80.14 `qa_Embedded` vs cpr: a core header was reaching into the HTTP stack (2026-09-01)

`qa_Embedded` builds `-fno-rtti -fno-exceptions` (the embedded/freestanding profile) and could not compile:
`cpr/body.h` needs exceptions. Chain was `qa_Embedded.cpp:18` -> `Graph_yaml_importer.hpp` ->
**`PluginLoader.hpp:30` (core)** -> `FileIo.hpp:36` -> `<cpr/cpr.h>`, guarded by `GR_HTTP_ENABLED`.

**The obvious fix is unsound — do not apply it.** `target_compile_definitions(qa_Embedded PRIVATE
GR_HTTP_ENABLED=0)` looks right and matches the AdaptiveCpp exemption three lines away in
`core/test/CMakeLists.txt:141-144`. But `libgnuradio-core.a` emits **101 symbols** for
`algorithm::fileio::readAsync`, and the `#if GR_HTTP_ENABLED` at `FileIo.hpp:1004` is _inside_ that `inline`
function's body. Per-target redefinition gives one inline function two definitions in one program — an ODR
violation resolved arbitrarily by the linker. It compiles clean and is wrong.

**Fix applied:** `PluginLoader.hpp` used exactly one thing from `fileio` (`readAsync`, inside
`readUriToString`, whose only callers sit in the same header in a non-template member). Moved that definition
into the already-existing `core/src/PluginLoader.cpp` and left a declaration. The header no longer mentions
`fileio`: one definition in one TU, no ODR hazard, and core consumers stop dragging in the HTTP/TLS headers
(related to the 5.5 MB unused HTTP stack in the UC3 footprint note).

Verified on all three: gcc15-debug 107 compiles / 96 of 97 (**first full completion of that config this
session**), acpp 100 of 100, registry 137 of 138. `qa_Embedded` itself passes 87 asserts / 15 tests.
Pre-existing failures unrelated to this: `qa_SoapyIntegration` (Timeout — zero references to PluginLoader,
hangs in a hardware-polling loop with no SDR device) and `qa_PythonBlock`.

#### 80.15 The block-registry build dir tolerates neither partial nor full deletion, nor interruption (2026-09-01/02)

Three distinct failures in one session, all from the generated `plugins/` tree, all looking like source defects:

- **partial `rm -rf build-*/plugins/<X>` + reconfigure** -> `undefined reference: gr_blocklib_init_unit_*` for
  _other, unrelated_ plugins. Fix: full reconfigure.
- **full `rm -rf build-*/plugins`** -> configure becomes UNRECOVERABLE (`Cannot find source file:
plugins/qa_grc/integrator.cpp`), because configure aborts at `target_sources` before the generator can
  regenerate those units. A second pass does not help. Fix: recreate the whole build directory.
- **a stale generated LINK LINE** -> a `.so` that lacks symbols its own compiled objects define
  (`gr_blocklib_init_unit_PeakDetector_0` present in the `.o`, absent from `libGrFourierBlocksShared.so`).
  ROOT CAUSE, corrected: `CMakeFiles/GrFourierBlocksShared.dir/link.txt` listed 6 objects (`fft_0..4`,
  `integrator`) and no PeakDetector, because the link line predates the generated unit. **When the block-lib
  generator's unit list changes, `cmake --build` does NOT regenerate link lines -- only an explicit
  `cmake <builddir>` does.** It is deterministic, not flaky: it recurred after a clean 1082-compile rebuild with
  no interruption. My first diagnosis ("interrupted build") was wrong. Fix: reconfigure, then rebuild.

**Only two operations are safe: a plain reconfigure, or recreating the directory.** Also: the generator does
not re-run on a header edit alone, so a registry check needs a reconfigure, not a rebuild.

**Reading build results:** `ctest` after a failed build silently runs the previous binaries and reports a clean
pass. Always gate ctest on the build's return code. And CMake's progress percentage is target-based, not
time-linear -- it badly under-reports the effect of raising `-j` (18%->38% at -j1->-j3 looked flat while actual
throughput went 1.8 -> 4.4 compiles/min).

**Shared machine:** another Claude session builds `~/temp/gnuradio4/build-bc-gcc16` on this box. `pkill -f
<pattern>` is process-wide and WILL kill their compiles -- never use it here; it also matches the running
script's own command line if that contains the pattern (`build-acpp` matched `acpp`), killing the job itself.
Watch swap: at 28 GB of 46 GB swapped, back off to -j1/-j2 rather than adding parallelism.

#### 80.16 Comment pass: why it is ONE commit, not per-commit fixups (2026-09-02)

`7b261952 style: say the non-obvious things and stop narrating the rest` -- density 7.5% -> 3.7% (961 -> 461
added comment lines), G10 12 copies -> 1, two dangling `claude_wip.md` references removed, prose blocks 46 -> 23.

**Per-commit distribution was attempted and abandoned for a structural reason, not for lack of effort.**
Hunk-level blame mapped all 138 hunks onto 11 on-branch commits and a splice round-trip reproduced the tested
tree byte-for-byte, but the autosquash conflicts irreducibly: **clang-format's re-alignment couples a comment
edit to code that LATER commits added**. In `qa_DeviceResidency.cpp` a hunk blamed to the device-runtime commit
carries context naming `mirrorsItself`/`sourcePort`/`resource`, fields that do not exist at that commit. Three
conflicts in, each hand-resolution risking a silently dropped change, one reviewable commit beat a distribution
built on manual merges. If this is retried: expect the same, and only per-hunk-per-file splitting that also
excludes format-coupled hunks could avoid it.

Tooling notes for a retry: derive hunks from a SAVED copy, not the working tree, and pin the base ref -- `git
show HEAD:f` moves as each fixup commits and silently absorbed 7 hunks into the wrong commits before I caught it.

#### 80.17 Precondition (4) DONE, and (5) measured rather than assumed (2026-09-02)

**(4) proven** — commit _"test(core): pin the state layout a fast convolution needs before building one"_. A
`std::pmr::vector<gr::complex<float>>` the block DERIVES in `settingsChanged` is seated on the device resource,
read from a relocated kernel with zero fallbacks, and re-derived when the source setting changes mid-run. It
only compiles because `gr::complex` became a supported settings type (§80.11) -- a reflected pmr vector of it
must be one.

Three things the probe corrected, each from a failing assertion, not from reading:

- **A derived member does NOT lose its device seat when reassigned.** I predicted it would, because
  `migrateFieldsToDeviceResource()` early-returns once seated. Measured: `resize` and `= std::move(fresh)` both
  stay seated, because `std::pmr::polymorphic_allocator` does not propagate on move-assignment. (`swap` still
  would -- that is the `HistoryBuffer::resize` bug this branch already fixed.)
- **Assigning the source field directly never fires `settingsChanged`**, so the derived member stays empty and
  the kernel reads nothing. `ScaleByTaps` cannot show this: it reads its pmr member directly and derives nothing.
- **A parameter given at construction is erased from auto-update** (`Settings.hpp:1007`), so it then ignores
  every later tag. A block with derived state must take its inputs through the settings system, and if it must
  also follow tags, it cannot be configured at construction.

**(5) — the blocker, measured.** `BasicFilterProto<float>`: `sizeof` 3648, `DeviceRelocatable = true` (the
§80.2 gap, live), `DeclaresDeviceStateReflected = false` (so the diagnostic warns), **but
`AutoParallelisable = false`** -- its `processOne` is non-const and its `processBulk` takes plain `std::span`,
not the view/span concepts. **So no tier accepts it today and it falls back safely; the hidden `_filter` becomes
dangerous at exactly the moment (5) makes the block device-capable.**

**History: template-static is required, pmr is actively wrong (measured 2026-09-02).** `copyBackUserState`
returns a member iff it is **trivially copyable and not `PmrMigratable`** (`DeviceRelocatable.hpp:158`):

| history form                                          | trivially copyable | PmrMigratable | returns from device |
| ----------------------------------------------------- | ------------------ | ------------- | ------------------- |
| `HistoryBuffer<T, N>` (template-static, `std::array`) | true               | false         | **yes**             |
| `HistoryBuffer<T, dynamic_extent, pmr_alloc>`         | false              | true          | **no**              |

The two cannot be combined: static `N` selects `std::array` and takes no allocator; asking for pmr forces
`dynamic_extent`. **Both** block shapes report `DeviceRelocatable = true`, so the pmr form is admitted and its
advanced indices silently never come back -- the §80.9 hazard, concretely. So a device IIR's history must be a
template-static `HistoryBuffer`, capped at compile time, and must NOT be seated.

Coefficients are the opposite case: read-only on the device, so a seated reflected `std::pmr::vector` is right
and `copyBackUserState` correctly skips them.

**Tier constraint, read from the code:** the auto-parallel tier (`ExecutionStrategy.hpp:698`) and the view tier
(`:642`) both REFUSE a body that mutates the block. The span tier does not -- one work item, no race. So a
stateful filter must use the span tier, as §80 predicted.

Why `_filter` cannot travel as-is, measured: `HistoryBuffer<float,8>` IS trivially copyable (static
`std::array` storage), but `detail::Section<float,8>` is NOT -- it inherits `FilterCoefficients` with
`std::vector<T> b, a` -- and `Filter` holds `std::vector<Section>`. **Two levels of host heap**, so a statically
sized `bufferSize` alone is not sufficient; the coefficients and the section list are the remaining ones.

#### 80.18 STALE-MIRROR CLOBBER — real, demonstrated, no instance yet (2026-09-02)

Surfaced by the fable review, then reproduced. **A span-tier block's out-of-band host state change is silently
overwritten by the stale device mirror.** Demonstrated end to end:

```
after dispatch 1          : host=11  (kernel advanced 10 -> 11)
host mutated out of band  : host=99  epoch 0 -> 0 (UNCHANGED, so the mirror is not refreshed)
after dispatch 2          : host=12  <-- the 99 is gone
```

The chain, each link verified:

1. a direct host-side write does **not** bump `settingsEpoch()` (measured: 0 -> 0);
2. `deviceMirror` relocates **only** `if (shadow.epoch != block.settingsEpoch())`
   (`ExecutionStrategy.hpp:250`), so the mirror keeps last dispatch's bytes;
3. `copyBackUserState` runs after **every** span-tier dispatch (`:541`);
4. so the stale mirror's state is written back over the fresh host value.

**The inconsistency, stated precisely:** the `shadow.epoch` cache in `deviceMirror` serves ALL tiers, but
`copyBackUserState` has exactly ONE call site (`:541`, the span tier). Caching a mirror across dispatches is
sound only while the mirror is read-only -- which is true for the auto-parallel and view tiers and false for the
span tier, where the mirror is authoritative between dispatches. The comment at `:251` ("read-only on the
device; nothing is copied back") describes the other two tiers and is wrong for this one.

**Reachable triggers** (mechanism verified, scenarios inferred): an interleaved CPU-fallback work call (the host
body mutates host state directly), lifecycle `reset()`, or any public method a user calls between dispatches --
for a filter block, `designFilter()`, which every existing test calls.

**No instance today**: no shipped block has mutable trivially-copyable state on the span-tier path
(`ZeroCrossingTrigger` in `qa_DeviceSpans.cpp:78` is the only span-tier stateful block and nothing mutates it
out of band). It becomes live with the filter migration or spike A2 -- same status as the §80.9 hazard.

**Proposed fix (not implemented, needs a decision):** refresh the mirror unconditionally whenever
`copyBackUserState` will run, i.e. gate the epoch cache on "the mirror is read-only". Cost is one
`relocateBlockToDevice` memcpy of `sizeof(TBlock)` per span-tier dispatch (3648 B for the filter block),
negligible beside a kernel launch, and idempotent -- last dispatch's state was already copied back into the host
object. Note `firstStaleMirrorMember` detects stale **pmr** members only; trivially-copyable state has no
staleness detection at all.

#### 80.7 Pre-commit progress (reviewer's decomposition, adopted 2026-08-30)

- [x] **(1) gate diagnostic — DONE**, commit _"fix(core): say so when a block may be keeping state the device
      cannot see"_, sits on top pending promotion into the frozen 20. **The planned size-coverage heuristic was
      built, measured and REJECTED**: `alignof(Block) == 64`, so legitimate padding reaches 63 bytes and swamps a
      24-byte hidden container — the probe's 80-byte reading was an accident of layout, and no principled
      threshold exists. Shipped the exact form instead: a block declares
      `using DeviceStateIsReflected = void;`, and `deviceMirror` reports the absence once per block **type**
      (function-local static, not a member — `sizeof(Block<T>)` untouched, one-layout ODR commit intact).
      Eligibility unchanged; a silent block still runs, it is merely audible. `processBulk_sycl` blocks never
      copy the object and so are never asked. Verified both ways on hardware: 7 types warned, 0 after declaring,
      assert counts unchanged; ctest 100/100 against the captured baseline.
- [x] **(2) `HistoryBuffer` allocator-awareness + `resize()` — DONE.** The swap bug was real: reverting it makes a counted resource underflow to `liveBytes = 2^64-96`, i.e. memory freed through a resource that never allocated it.
- [x] **(3) enum-for-string swaps — DONE** (`fft.hpp` window, SavitzkyGolay alignment + boundary policy; both hand-rolled conversions silently defaulted on a typo, and boundary policy could only reach 2 of its 4 values)
- [ ] (4) prove the device state layout greenfield in the new fast-conv block
- [ ] (5) migrate the tested filter blocks onto the proven layout — **BLOCKED BY R0**: unbuildable as posed,
      `Section`/`HistoryBuffer` are exactly the shape with no state category

**Baseline for judging regressions: ctest 100/100 at `b9a1cba9`** (per-test outcomes captured). Note the
block-level `qa_FFT` / `qa_filter` suites are configured in NONE of the three build dirs, so they are outside
that baseline — `qa_FilterTool` (the oracle) and `qa_buffer` (HistoryBuffer) are inside it.

**Reviewer's proposed re-decomposition (contradicts the single pre-commit — maintainer decision needed):**
(1) the §80.2 gate diagnostic first, as the safety net; (2) HistoryBuffer allocator-awareness + the `resize` fix
alone; (3) enum-for-string swaps alone (verified genuinely safe); (4) prove the device state layout greenfield in
the new fast-conv block; (5) only then migrate the tested filter blocks onto the proven layout. Rationale: the
current plan does the highest-risk step against the most-tested code first, with the least device feedback, and
its failure modes are Release-silent.

#### 80.1 There is already a filter layer — extend it, do not duplicate

Found 2026-08-30, before writing anything:

- **`algorithm/.../filter/FilterTool.hpp`** (1076 lines) — `FilterParameters`, `FilterCoefficients<T>`,
  `Type{LOWPASS,HIGHPASS,BANDPASS,BANDSTOP}`, `Form{DF_I,DF_II}`, `Section<T,bufferSize>` biquads, and a
  `constexpr computeFilter(input, section)` IIR core already generic over `arithmetic_or_complex_like T`.
  It also mocks `std::execution` (libc++ lacks it) — the device path cannot use it either.
- **`blocks/filter/time_domain_filter.hpp`** (249 lines) — `BasicFilterProto<T, Args...>` and a `Decimator<T>`
  carrying `Resampling<1,1,false>`.

Consequences for the plan:

- **`computeImpulseResponse` is a ready-made numerical oracle** — impulse response is the canonical filter test.
- `Section<T, bufferSize>` defaults `bufferSize = std::dynamic_extent`, which allocates. The device path needs
  the **statically sized** variant, or the block is not `DeviceRelocatable`.
- So spike A is largely _make the existing core device-callable and add a frequency-domain path_, not write a
  filter from scratch.

### READING KEY — read this before quoting anything below it

1. **`Domain` is `SubGraph`.** Anywhere below that says `gr::device::Domain`, `makeDomain()` or
   `device/DeviceDomain.hpp`, today's code says `gr::SubGraph`, `makeSubGraph()` and `core/.../SubGraph.hpp`. The
   rename was folded into the rebuilt history, so the old name appears in **no commit** — only here.
2. **The GLSL and WebGPU backends left core** under decision D5 (§50, §55, §57). §8, §20-§21, §26-§27, §29 and
   §31-§34 describe code this branch no longer contains — kept for the reasoning, not as a description of the
   tree. **§25 and §35 are different: they are the measurement record**, and the numbers in them (together with
   §53 and `reference_fft_backend_benchmark_table`) are what D5 rests on. Do not treat those two as obsolete.
3. **Removed API still quoted below:** `hostTransferCount()` / `hostUploadCount()` / `hostDownloadCount()`,
   `copyDeviceToDevice()`, `shaderFragment()`, the chain/epoch machinery, and the `qa_FftCrossover` harness.
   §68 is the standing note on what has to come back, and why, before device-to-device residency can be built.
4. **Two things are out of scope by maintainer decision (2026-08-25), not merely deferred:** sub-graph
   auto-formation (§38 — a helper, no core change, a later PR if wanted) and `processEpilogue` on the device
   path (a CPU concern: SIMD batch sizes and tag-split frames, neither of which a device has). Anything below
   that lists either as open is stale.

### What this branch is now

A **`gr::SubGraph`** (`core/include/gnuradio-4.0/SubGraph.hpp`) is a group of blocks that one in-built scheduler
drives synchronously, from its own `work()`, on the caller's thread. `makeSubGraph()` takes an ordinary sub-graph
and exports every port no interior edge claims. It is plain core infrastructure — after the shader chain machinery
was removed it references nothing device-specific, which is why it lives in core and not under `device/`.

`makeDeviceSubGraph<T>()` (`blocks/basic/.../DeviceSubGraph.hpp`) is the device-facing wrapper: it inserts a
`HostToDevice<T>` in front of every unclaimed input and a `DeviceToHost<T>` after every unclaimed output, then
exports the transfers' outer ports. Membership declares the boundary, so no transfer is ever inferred.

**Two backends, and that is the whole list:** host CPU and SYCL. `compute_domain` (§50.11) selects between them —
one reflected string, grammar `kind[:backend[:deviceIndex]]`, doubling as CPU thread-pool name and device selector.

**The honest verdict, unchanged: this ships as a capability, not as a speed-up.** §53 is the quantified reason —
the GPU only wins the FFT at N >= 4096 _and_ batch >= 16, and loses at every size at batch = 1. What the branch
buys is that a graph moves to a device by changing a string rather than a type.

### Where to read, in order

`§50` the functional envelope and the D1-D11 maintainer decisions · `§50.11` what `compute_domain` actually means ·
`§53`/`§54` the FFT and FIR litmus tests — the evidence behind dropping the shader backends · `§55` the branch
re-scored against the original premise · `§57`-`§58` the D5 drop and the history rebuild · `§60`-`§64` phase 5 ·
`§66`-`§67` the D8 host-boundary measurement · `§68` device-to-device, and what it needs.
**§15 gotchas G1-G22 are hard-won — read before touching device code.** G10 in particular: a device test must be
driven from `main()`, never from a Boost.UT global suite.
Backend performance numbers also live durably in Claude's memory as `reference_fft_backend_benchmark_table`.

### The 20 commits

```
20 feat(core): keep an interior device edge in memory the host cannot address
19 perf(core): give a device edge that crosses to the host pinned memory, not shared
18 fix(core): make the single-device-domain rule a property of a running group
17 feat(basic): insert a group's host/device transfers instead of hand-wiring them
16 test(core): the vertical stack a device sub-graph is meant to carry
15 build,ci,docs(device): compile the stack only when asked, and prove CI ran it
14 feat(fourier): one FFT that runs on the host or a device
13 feat(basic): move samples across the host/device boundary explicitly
12 feat(device): diagnostics from inside a kernel, rendered on the host
11 feat(core): run a group of blocks as one scheduling sub-graph
10 feat(core): dispatch a block to a device without changing how it is written
 9 feat(device): a device runtime behind one backend-neutral contract
 8 feat(core): express memory and execution that are not the host's
 7 fix(core): publish a tag without a concept enumerating the map types
 6 fix(core): read a wire map's values without materialising one
 5 fix(core): let a wire map nest another without allocating
 4 fix(core,algorithm): stop three tests failing on things they do not test
 3 fix(sdr,timing): stop discarding results that report a failure
 2 fix(core): keep a moved block's memory resource
 1 fix(core): release both halves of a double-mapped buffer
```

Commits 1-7 are pre-existing `main` defects found on the way and kept disjunct, so they can be cherry-picked
ahead of the feature work — 1-4 found while building, 5-7 the wire-map defects that the device tag path
exposed but did not create. (Two commits in the old 16-item list, `refactor(core): let LogRecord fill its own
fixed arrays` and `refactor(device): drop the transfer counters and device-to-device copy`, were folded into
the commits they served; the wire-map and test fixes were added at the front.)

**Read `### NEXT` below before picking up any work.** The tables under it are current as of 2026-08-28; every
numbered section further down the file is a record, not an instruction.

### ➡️ NEXT — nothing queued (maintainer, 2026-08-30)

**N1 is DONE and squashed into commit 10.** All four dispatch tiers take the ports a block declares; residency is
per port; the CPU fallback serves every shape including multi-port sources and sinks; the self-mutation probe
synthesises one sample per port; tag-storage reservation failure is reported rather than silently dropping tags.

**Dropped by maintainer decision — do NOT re-add to a ToDo list:**

- **N1's residual test coverage** — mixed-residency never asserted, no empty/single-sample case at N-ary arity,
  the N-ary canary only exercised through a direct call rather than a graph, partial allocation failure unforced.
- **N2 — group-to-group device chaining** (hoisting a group's shared `compute_domain` onto its exported ports,
  Route A, §73). Decided and unbuilt.
- Reason given: their use cases and failure modes are rare. Within a group and in a flat graph device-to-device
  already works; anything needing it across a group boundary can be one larger group.

The branch is feature-complete for this PR. Remaining work is review and merge, not implementation.

### 🏗️ DESIGN EXCURSION — stride (maintainer, 2026-08-31)

_(The port-collection half moved to `~/gr4-snapshots/port_collection_intent.md` on 2026-09-08.)_

**Stride works; an earlier note here doubting it was wrong.** `qa_Block`'s _Stride Tests_ already cover both
directions — skipping (`stride > chunk`) and overlap (`stride < chunk`, e.g. `chunk=100, stride=50` -> 19 calls,
`total_in=1900`, a 1.9x re-read). Re-verified at the FFT's own parameters (`n=128, chunk=16, stride=8` -> 15
calls, 240 in/out): passes. The streaming FFT now declares `gr::Stride<>` alongside its `Resampling<>`, matching
`Resampler` in that suite.

**Correction to an earlier note in this file:** the claim that overlap-save "cannot be expressed" is **WRONG**.
`stride` is a first-class setting on every block (`Block.hpp:735`, doc: _"<N for overlap, >N for skip, =0 for
back-to-back"_). The base defaults to `Stride<0UL, true>` — `isConst=true` makes `kEnabled` false, which is why an
undeclared block refuses a non-zero stride. Declaring `gr::Stride<0U>` makes it runtime-settable at the same
default; precedent: `FrequencyEstimatorFrequencyDomain<T, Resampling<1U>, Stride<0U>>`.

### OPEN — re-verified against the code 2026-08-28

| #   | item                                                                                                                                                                                                                                                                                                     | evidence                                                                                                                                                                                                                                                                                               | blocked on                                                                                                                                                                                 |
| --- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 1   | `makeDeviceSubGraph` takes one `T` for every boundary                                                                                                                                                                                                                                                    | §64 — a guard was written, segfaulted and was abandoned. **The old "silently gets a wrong-typed transfer" wording was WRONG:** `Graph.hpp:793` marks the edge `IncompatiblePorts` and `Scheduler.hpp:769` logs an error, then proceeds, so the port fails to wire loudly and keeps its default buffer. | Route B (§73.4) — inherit the inner port's `metaInfo`; root-cause the §64 segfault first rather than retrying it                                                                           |
| 2   | **No sanitized lane reaches device dispatch.** `ExecutionStrategy.hpp` is included only under `GR_DEVICE_HAS_ANY_BACKEND`, which needs `__ACPP__`, and AdaptiveCpp cannot be built with ASan/UBSan/TSan. The ASan lanes reach the SubGraph and the transfer blocks (both backend-free) and nothing else. | verified 2026-08-28; stated in `ci.yml` and in commit 15's message                                                                                                                                                                                                                                     | a CPU `DeviceContext` that compiles without AdaptiveCpp would close this **and** the `gpu:sycl` CI-skip gap below                                                                          |
| 3   | A `gpu:sycl`-gated test still skips on a GPU-less runner, so the device-only interior edge (the 2.2x) is verified on hardware only. `GR4_REQUIRE_DEVICE=host:sycl` catches "SYCL stopped coming up", not "the GPU test skipped".                                                                         | `CMakeLists.txt:284`; `qa_DeviceResidency` gates on `firstServedDomain({"gpu:sycl"})`                                                                                                                                                                                                                  | correct as designed — a runner without a GPU cannot be made to have one. Stated in the PR                                                                                                  |
| 4   | **F1** — device-side mutation of a captured block as a compile error                                                                                                                                                                                                                                     | needs C++26 reflection                                                                                                                                                                                                                                                                                 | the standard                                                                                                                                                                               |
| 5   | Nothing counts transfers, so elision claims are structural rather than measured                                                                                                                                                                                                                          | the counters went out as dead API                                                                                                                                                                                                                                                                      | rebuild the counter FIRST if in-place chain elision is revisited — it is the instrument every mechanism choice depends on                                                                  |
| 6   | Blob alignment: `ChunkBuffer.hpp:153` starts every chunk's first blob at `sizeof(SealedHeader) = 24`, and `24 mod 16 = 8`; `:160` adds no rounding. Not merely unguaranteed — deterministically misaligned.                                                                                              | §74.1 corrected the number, sharpened 2026-08-28                                                                                                                                                                                                                                                       | **the tag work, not this branch.** Safe here by construction: `ExecutionStrategy.hpp:373` stages every device-bound tag blob into an aligned slot, and the other three tiers touch no tags |
| 7   | In-place elision across a chain: N device blocks hold N device rings, not one shared buffer                                                                                                                                                                                                              | §68.0                                                                                                                                                                                                                                                                                                  | a _frugality_ item, not a capability gap; gated on item 5                                                                                                                                  |

### CLOSED 2026-08-27/28 — all verified, none inherited

- **ctest 100/100** (was 97). Three suites were failing on things they do not test: a build option they never checked
  for (`qa_SubGraphAssets`, whose five tests burned two 120 s wall-clock caps each and read as a hang — **it never
  hung**; `qa_Graph`), and the argmax of a monotonic step (`qa_DataSetEstimators`). One commit at position 4.
- **Device-to-device fan-out is now TESTED.** The original evidence was `scratchpad/fanout_probe.cpp`, which is not
  in the tree. `qa_DeviceResidency` gained a host test (both consumers fed from one port) and a `gpu:sycl` test
  (both arms device-only, naming the **same resource**). Mutation-tested on hardware.
- **The mutation canary now covers pmr-backed blocks.** `DeviceProbeSafe` is simply `DeviceRelocatable`;
  `mutatesItsOwnState` captures the shared pmr bytes and restores them, so the probe is an observation again.
  Container writes stay unreported (shared with the device, nothing is lost); it declines only for a member seated
  on a device resource. The defect was the _mixed_ block — one pmr setting disqualified the whole block, so a
  trivially-copyable `mutable` sibling's discarded write went unreported. That is the FIR shape.
- **The span tier has separate in/out counts.** Worse than "output bounded by input": an interpolator never met its
  resampling ratio and **the graph hung** (reverting times `qa_DeviceSpans` out at 300 s). GLSL/WebGPU legacy — a
  shader dispatch has one fixed grid, so a variable rate had nowhere to live.
- **Two false claims about our own coverage removed** — the `ci.yml` comment and commit 15's message both said a
  sanitized lane exercised `ExecutionStrategy`. It cannot: that header needs `__ACPP__`.
- **Identity normalised**: all 20 commits author/committer/sign-off `Ralph J. Steinhagen <r.steinhagen@gsi.de>`.
  No Claude/Co-Authored-By references existed in any message. Tree hash unchanged through every rewrite.
- Earlier: transfer insertion for a group, the single-domain invariant, D8's pinned host boundary, the dead-API
  drop, the review cycle (§40-§46), the P2300 extraction, the `Domain`→`SubGraph` rename, the de-shadering sweep
  (§70), and the `qa_FftCrossover` harness (a debugging aid; the crossover **numbers** came from §53 and
  `reference_fft_backend_benchmark_table`, both intact).

**Both durable memories that contradicted this branch have been corrected** (2026-08-27/28):
`project_gr4_device_branch_state.md` records the current commit count, the SubGraph rename, D5 and the sweep, and
`project_gpu_fft_streaming_bottleneck.md` leads with staging (15x) rather than pinning (1.17x), per §66.1.

### ✅ TODO LIST — BATCH COMPLETE (2026-08-08)

Tiering used: Opus coordinating + medium work · 1 Sonnet agent for the mechanical pair · 1 Fable agent pre-assessing the two
ambiguous items, with the one genuine policy choice escalated to the maintainer as multiple choice. Commits were
pre-authorised **for this batch only** — the ask-every-time order resumes now.

| #   | item                                   | state                                                                                                                                                                                                                                                                                                                                                                                                   |
| --- | -------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | rebase onto present `origin/main`      | **DONE — no-op.** `origin/main` still `92278b62`, merge-base equals its tip, 0 behind, 0 conflict surface. No push, no PR.                                                                                                                                                                                                                                                                              |
| 2   | demote the per-`dispatch()` entry sync | **DONE** (`6143c36c`). `peekDeviceError()` added: waitless `throw_asynchronous` drain + latch read; entry call swapped. Trailing polls untouched, and the `dispatchSyclBulk` barrier (`:206`) deliberately left — it is what makes an async hatch's output valid before publish. Sound only while framework dispatch is internally synchronous; stated in the commit message.                           |
| 3   | rename the colliding registry          | **DONE** (`6143c36c`). `gr::device::SchedulerRegistry` → **`DeviceContextRegistry`**, file renamed, 13 files updated, 0 stale refs, `gr::SchedulerRegistry` untouched.                                                                                                                                                                                                                                  |
| 4   | `dispatchTiers` → private              | **DONE** (`6143c36c`).                                                                                                                                                                                                                                                                                                                                                                                  |
| 5   | `Block<T>` one layout in every config  | **DONE** (`5115c1e8`, own commit as instructed). Maintainer chose _always carry the 32 bytes_. Members unconditional, only the code guarded; needed one light include (`DeviceBlockShadow.hpp`, backend-free). **Proven:** `sizeof(Block<P>)` = 2752 with the backend OFF (gcc15) and ON (acpp), previously divergent.                                                                                  |
| 6   | compiler matrix                        | **DONE, all four legs green.** gcc15 CPU-only (GL now OFF): qa_buffer/qa_Graph/qa_Block/qa_Scheduler/qa_PerformanceMonitor · clang20 libc++: qa_buffer/qa_Graph/qa_Block · acpp (SYCL+GL): qa_DeviceBlockStyles/qa_DeviceAutoParallel/qa_DeviceErrorChannel/qa_DeviceResidency · wasm **option (b)**: TU gate over every touched header with the project's own Debug flags + full `-Werror` set, clean. |

**Tip `5115c1e8`, 45 commits above `origin/main`, nothing pushed.**

**Follow-ups this batch created (do not lose):**

- `qa_DeviceErrorChannel` tests the context API directly and would **NOT** catch a dispatch-entry regression. Extend it: assert `peekDeviceError()` reports a latched fault, and drive a block through `dispatch` asserting the "device context poisoned" fail-fast.
- Write down the rule: **any code submitting device work outside `ExecutionStrategy::dispatch` must do its own trailing `pollDeviceError()`**, or its fault surfaces only at the end of the next dispatch. The residency-by-edge-domain direction will create such a caller.
- If the framework's internal `.wait()`s are ever dropped, redo the entry-poll barrier analysis per edge.
- `_settingsEpoch` stays unconditional deliberately (8 B, generic call sites, host half of the mirror-coherence protocol) — gating it would reintroduce the layout `#if` item 5 just removed.

### ⚠️ FOUR FOLLOW-UPS FROM THE FINAL GOAL REVIEW (2026-08-08) — none started

1. ~~**GL CI lane**~~ **— DONE 2026-08-09, and it uncovered three bigger things.** See the block below.

### ✅ GL CI LANE + CPU-SYCL STAND-IN — LANDED AND COMMITTED 2026-08-09 (the '(uncommitted)' in the old heading was stale)

**The acpp CI lane was already RED on this branch.** It exists on `origin/main` unchanged (`ACPP_TARGETS=omp`, bare
`ctest`, no label filter) and every `qa_Device*` target is branch-new, so all of it starts running on a GPU-less runner
the moment this merges. Reproduced locally with `ACPP_VISIBILITY_MASK=omp` (hides the GPU, keeps the OMP SYCL device —
the cheapest way to get the CI situation without a rebuild; remember this trick):
`qa_DeviceSeam` **SIGSEGV** (unguarded `sink._tags[0]` after a failed size expect — the CPU fallback drops the tag),
`qa_DeviceAutoParallel` **2 of 4 FAILED**, plus silent skips (`qa_DeviceResidency` 14→6 asserts, `qa_DeviceLoggerBackend`
18→9, `qa_DeviceBlockStyles` 16→11). After the fix every binary is rc=0/77 with the GPU hidden; `qa_DeviceSeam` 12/1,
`qa_DeviceAutoParallel` 26/4 (30/4 with a GPU), `qa_DeviceBlockStyles` 15/6.
**`GlComputeContext::init()` could not create a context in a GPU-less container at all** — simulated with
`unshare -rm` + `/dev/dri` masked + Mesa-only glvnd: `eglInitialize` → `0x3001`. So a GL lane on the old code would have
been green and covered nothing. Fixed: prefer `eglGetPlatformDisplayEXT(EGL_PLATFORM_SURFACELESS_MESA, …)` (gated on the
client extension string — `eglGetProcAddress` returns a stub for entry points glvnd does not have), accept
`EGL_NO_CONFIG_KHR` when no `EGLConfig` matches, and warn on every failure path instead of returning silently.
Container sim now gets **llvmpipe GL 4.6 core** with assert counts identical to native.
**Every GL number recorded before today was llvmpipe, not the GPU.** `eglGetDisplay(EGL_DEFAULT_DISPLAY)` fell into Mesa
even with `DISPLAY=:0` and an RTX 3070 present (Mesa fails dri2 on the NVIDIA card and falls back to swrast). The fix
makes the same call reach the real driver. The backend context now **logs its renderer** so this cannot recur silently —
that one missing log line is what let a software rasteriser masquerade as `gpu:glsl` for the whole effort.
NOTE: the **authoritative streaming table above is `gpu:sycl` and is unaffected**; only the `bm_FFT_backends` GLSL column
was software. Re-measured best-of-3, clocks pinned 1500 MHz, `GLSL:NVIDIA GeForce RTX 3070` self-reported:
GLSL peaks **27.9 GFLOP/s** (x128, N=16384) against SYCL-GPU **78** and SimdFFT **15.7**; at x1 GLSL is 0.2–12,
i.e. GL only earns its keep at large batch. (x128/N=65536 GLSL reads 12.6 — an outlier worth one look, not chased.)
**Do NOT set `EGL_PLATFORM=surfaceless` in CI** — it routes to a vendor with zero `EGL_OPENGL_BIT` configs and breaks
context creation. The lane needs no env vars at all.
Also landed: `DeviceContextRegistry::resolve()` **deleted** (silent CPU fallback with it; no production caller ever
existed), `firstServedDomain()`/`firstServedSyclDomain()` test helpers, `GR_ENABLE_GL_COMPUTE=ON` now **FATAL_ERRORs**
without pkg-config instead of silently staying off, `libgl1-mesa-dri` + `pkg-config` added to the container,
`qa_FftCrossover`'s two wall-clock bands moved behind `DISABLE_SENSITIVE_TESTS` with the plumbing check split out
untagged, and a missing `<functional>` in `execution.hpp`. (`qa_FftCrossover` itself was dropped 2026-08-25 — a
debugging aid, not a regression test; the crossover figures it produced are in §53.)
**From the 2026-08-09 review (both resolved since):**

- ~~**the CPU-fallback path drops a published tag** — bug or intended?~~ **DECIDED 2026-08-22, intended (§36.1):**
  only `gr:`-prefixed and explicitly opted-in keys are forwarded, and per-sample tag handling in `processOne`-only
  blocks is being discouraged. The test block's custom key was outside the forwarded set by design. **Closed.**
- ~~GL-gated files get `-Werror` coverage on gcc-15 only~~ **CLOSED 2026-08-09**: the **AdaptiveCpp lane now also sets
  `-DGR_ENABLE_GL_COMPUTE=ON`**. Verified that `-Werror` really reaches that lane — the acpp compile line carries
  `-Wall -Wextra -Wconversion -Wshadow -Wold-style-cast -Wsign-conversion -Wpedantic -Wdouble-promotion -Wcast-align
-Wnull-dereference -Wformat=2 -Werror` (`cmake/CompilerWarnings.cmake:53`; acpp is **not** excluded anywhere), and
  acpp is a **Clang-21 frontend**, so it is the closest thing to Clang-20 diagnostics the matrix has. Compiled clean
  under acpp `-Werror` after touching the headers: `GlComputeContext`, `DeviceContextGLSL`, `GlslRuntime`, `GlslFFT`,
  `bm_FFT_backends_helpers`, `device_test_helpers`, `qa_DeviceContext`, `qa_DeviceBlockStyles`, `qa_DeviceMultiplyConst`;
  `qa_FFT2` + `qa_FFT2Performance` pass `acpp -fsyntax-only` with the same flags (they are not build targets locally
  because `build-acpp` has `GR_ENABLE_BLOCK_REGISTRY=OFF`, which drops `blocks/*/test` — on CI the registry is ON, so
  they do compile there). Under gcc-15 + GL, `qa_FFT2` builds, links and gives **identical assert counts on the real
  NVIDIA context and on llvmpipe in the container sim**. GL is therefore compiled by **both** front ends in CI.
- `firstServedSyclDomain()` trap worth a comment: a binary that registers a _mock_ context named `gpu:sycl`
  (another suite does, in its own binary) and then calls the helper would assert against the mock and pass.
- `isDeviceAccessible()` accepts only `usm::alloc::{shared,device}`, so **host-pinned USM never elides a copy** — may
  matter for the staged-transfer work. Not touched here.

2. **F1 must become a compile error before merge.** Making GL an explicit opt-in fixed goal 4 (zero cost for CPU-only)
   but broke **goal 6**: `GR_ENABLE_GL_COMPUTE` defaults OFF, so the GL backend now has **zero CI coverage**. Add a lane that
   configures with `-DGR_ENABLE_GL_COMPUTE=ON` on Mesa **llvmpipe** (no GPU needed) and runs the GL-gated tests
   (`qa_DeviceContext`, the `gpu:glsl` cases in `qa_DeviceBlockStyles`, `qa_FFT2`'s GLSL leg). Small, and it should land with
   or before the GL PR (`PR-D`) — do not ship the opt-in without it.
   2b. **F1 must become a compile error before merge.** Device-side mutation and tag publication are **silently dropped** on the
   auto-parallel path: the block is memcpy'd into device memory and never copied back, so a mutated member or a
   `publishTag()` on the device copy vanishes, and `mergedInputTag()` returns a view aliasing host tag-ring memory that is UB
   to dereference in a kernel. So goal 1's "runs unchanged" holds **only for genuinely pure const blocks**. There is a
   debug-only guard today (`blockMutatesItsOwnState`, `ExecutionStrategy.hpp` auto-parallel path) — that is not enough. Make
   it a hard compile error (no `mutable` members, no tag access on the auto-parallel path) rather than silent data loss.
   This is a correctness gap, not polish. Tracked as F1 in §18.2.
3. ~~**`ShaderFusion.hpp` is still delete-or-justify.**~~ **RESOLVED 2026-08-09 — KEEP, with `GLSL2WGSL.hpp`.** The Re4 mandate re-legitimises **`GLSL2WGSL.hpp` — KEEP** (though it is
   a 115-line sketch that Re4 will likely replace rather than extend). It does **not** cover `ShaderFusion.hpp` (156 lines,
   test-only, F11 open): the mandate names GLSL/WASM, not kernel fusion. Keep it **only** if Re4's plan gives it a real
   graph-pass caller; otherwise delete and let git history hold it.
4. **Use this PR opening paragraph** (drafted in the goal review; honest about what this is without underselling it):
   > This series adds a heterogeneous-compute capability layer to GR4, not a performance feature. A block written as ordinary
   > `const noexcept processOne`/`processBulk` runs unchanged and correct on a SYCL or GL-compute device selected by the
   > existing `compute_domain` string, with typed native escape hatches (`processBulk_sycl`, `shaderFragment`) for experts, a
   > composed `ExecutionStrategy` behind a single seam in `Block<T>` (one layout in every configuration), an explicit error
   > channel for device faults, and edges that can live in memory the host never touches. Measured honestly — warm, clocks
   > pinned, interleaved — the streaming device path does not beat the host CPU across a host boundary (best case 0.83x at
   > 32 MiB edges; the wall is transport, not kernels, which occupy 0.5% of wall time); the one measured win is 1.45-1.47x on
   > a device-only edge between two adjacent device blocks, which is the seed of the on-device-chaining model the design
   > targets. Along the way the effort fixed a real pre-existing double-mapped buffer leak (cherry-pickable to main), killed
   > ten plausible performance claims by measurement, and leaves behind the harnesses and protocol to keep future claims
   > honest. Follow-up tracks: FFT2/FFT consolidation, and a WebGPU-grade shader
   > backend for the browser.

### 🧬 CARRY-OVER FROM THE WebGPU BRIEF (analysed 2026-08-13; brief = untracked `webgpu_mvp/webgpu_design_brief.md`, never commit that dir)

Judged against GR4's constraints, not the brief's framing. Each item had to remove a **present-day** defect, not future-proof.

**VERDICT ON THE GLSL DRAFT (2026-08-13): REPLACE, PHASED — it can never reach the browser.** Shaders are `#version 430`

- SSBOs (`ShaderFragment.hpp:45-48`, `GlslFFT.hpp:165-169`) and the context demands GL 4.3 core
  (`GlComputeContext.hpp:138`). WebGL2 is GLES 3.0: compute shaders and SSBOs only arrived in ES 3.1, the WebGL2-Compute
  experiment was abandoned for WebGPU, and `CMakeLists.txt:210` compiles GL out under Emscripten anyway. So "production-
  grade GLSL for the browser" is unreachable **by construction**, not merely unfinished. WGSL/WebGPU is the only dialect
  that runs in a browser _and_ natively (Dawn/wgpu). **Priority order is WASM/browser FIRST, native second.**

* **Do NOT build the brief's C++→SPIR-V→Tint compiler** — its own estimate is 8-20 person-months (brief §20). Ship hand-
  or framework-generated WGSL fragments first. (This corrects my earlier framing that SPIR-V→Tint was the near path.)
* **Deletes ≈1,060 lines** (GlComputeContext 314, DeviceContextGLSL 152, ShaderFragment 64, ShaderFusion 156,
  GLSL2WGSL 115, GlslRuntime 42, GlslFFT 215) plus the GLSL arms of ExecutionStrategy.
* **SURVIVES the replacement — a WebGPU backend is a NEW LEAF, not a rewrite:** the whole `ExecutionStrategy` tier
  architecture (expert hatch, named warn-once fallback, poisoned-context peek, mutation canary, stale-mirror
  diagnostic), `ComputeDomain` selection (`gpu:webgpu` is just a string — nothing in `Block.hpp` changes), the explicit
  opt-in build gate and its zero-tax discipline, residency elision by edge, and `GlslFFT`'s Stockham staging (ports to
  WGSL nearly verbatim; only the dialect and the baked `nBatches` need fixing).
* **Dependency reality:** Dawn would be the heaviest dependency GR4 ever carried (large non-header-only CMake tree;
  wgpu-native needs Rust or prebuilt binaries). Acceptable ONLY behind a `GR_ENABLE_WEBGPU` opt-in mirroring the GL
  gate, never default, never in the header-only core path. Browser side is near-free via Emscripten's emdawnwebgpu.
* **Defects worse than first reported:** `deallocateRaw` (`DeviceContextGLSL.hpp:110-118`) treats EVERY nonzero
  low-32-bit pointer as an SSBO id, so heap pointers from `allocateHostRaw` are passed to `glDeleteBuffers` as garbage
  and leak — wrong for any host allocation, not merely "no caller today". And the `GLSL2WGSL` test
  (`qa_DeviceContext.cpp:275-292`) greps substrings and never compiles its output, which is why it passes while the
  transpiler emits `uniform u32 uCount;` — invalid WGSL produced from its own generator's declaration.
* **PR-B CONSEQUENCE, supersedes the "typed handle before PR-B" plan above:** the better move is to REMOVE GLSL from the
  frozen surface rather than generalise the surface to accommodate it. Either drop the GLSL context from PR-B (the
  SYCL/CUDA pointer world is self-consistent without it) or keep `void*` plus a predicate gating the framework tiers off
  token-based backends — which fixes the live segfault at `ExecutionStrategy.hpp:433` / `ParallelFor.hpp:26` directly.
  **Do not freeze `allocateSharedRaw`-returns-token as API.** Also cut `GLSL2WGSL` + `ShaderFusion` (and their qa cases)
  from the re-cut: no dispatch-path caller, never tested in anger.
* **Next:** spike emdawnwebgpu in the browser behind `GR_ENABLE_WEBGPU` with ONE framework-generated element-wise WGSL
  kernel driven through the existing hatch pattern, proving the tier architecture is dialect-neutral, BEFORE any native
  Dawn commitment.

**BROWSER VALIDATION OF THE PoC — MEASURED 2026-08-13, not assessed. It runs, on two engines, unmodified.**
WASM build clean first try, **no source changes and no `--use-port`**: the PoC talks to `navigator.gpu` directly through
`EM_ASYNC_JS` glue rather than the emdawnwebgpu C++ port (`emcmake cmake -S . -B <dir> && cmake --build <dir> -j6`,
emcc 5.0.2). **Chrome** (headless=new, `--enable-unsafe-webgpu --enable-features=Vulkan --disable-gpu-sandbox
--no-sandbox`): full compute, all three graph runs **bit-exact** vs the CPU reference. **Firefox** (headless, profile
with `dom.webgpu.enabled=true`, `gfx.webgpu.force-enabled=true`): full compute, matching within float32 rounding
(worst `max_abs_error = 1.19e-7`). So browser-first is demonstrated, not merely plausible.
**The Dawn dependency objection is SOFTER than the verdict above assumes:** this MVP has **no Dawn/wgpu-native
dependency at all** — every WebGPU path sits behind `#ifdef __EMSCRIPTEN__` with stubs otherwise, and the native build
just runs the CPU reference and reports `webgpu: unavailable in this JS runtime`. The browser path costs nothing;
native WebGPU is _unimplemented_, not expensive. The Dawn question is deferred, not answered.

**STRUCTURAL FINDING — the WASM-heap ↔ GPUBuffer copy is mandatory, not a slow path.** The PoC's own log states it:
_"span probe: WebGPU completed via explicit buffer upload/readback, not by dereferencing a GPU pointer"_, backed by
`HEAPF32.slice()` on upload and `HEAPF32.set()` on readback. There is **no USM equivalent in a browser** — the sandbox
forbids it — so **`std::span::data()` can never resolve to a device pointer** the way it does under SYCL. Same problem
class as the 38 ns/sample host read-back already recorded, except unfixable rather than merely expensive.
**This settles the `ViewLike` design from a second direction:** a view whose `data()` is meaningful on-device is a
SYCL-only notion; the browser backend needs the contract to be **"a handle you copy through"**, not "a pointer you
index". Design the buffer contract handle-and-copy FROM THE OUTSET rather than retrofitting a pointer abstraction.

**TRAP, and it is the llvmpipe mistake again.** Headless Chrome reported adapter `vendor=google
architecture=swiftshader` **despite `--enable-features=Vulkan`** — a software rasteriser, not the box's RTX 3070. With
`--use-angle=vulkan --ignore-gpu-blocklist` the Chrome GPU process _did_ allocate on the real GPU (confirmed via
`nvidia-smi --query-compute-apps`, PID matched) yet Dawn's WebGPU adapter **still self-selected swiftshader** — ANGLE's
backend choice and Dawn's are independent. **Any performance figure from a headless box is a SwiftShader figure unless
the adapter string is asserted on every run.** (Chrome's bit-exact agreement with the CPU reference is consistent with
both sides running the same CPU float32 path.) Firefox masks adapter info by default (`vendor=<unknown>`), and a
`RenderCompositorSWGL failed mapping default framebuffer` line suggests the software compositor is at least involved —
so its backend is _unidentified_, not confirmed hardware. Exactly the failure mode as `eglGetDisplay` quietly landing on
llvmpipe and making every "GPU GLSL" number software.

**Harness notes worth keeping (cost ~10 turns to rediscover):** `--dump-dom --virtual-time-budget=N` **truncates the run**
— virtual time does not wait on real async GPU IPC — so poll via CDP `Runtime.evaluate` after a real wall-clock wait.
Firefox 153 headless `--remote-debugging-port` speaks **WebDriver BiDi, not CDP**: there is no `/json` endpoint (404),
the WebSocket handshake must target **`/session`** (root returns an httpd.js splash with HTTP 200 and never upgrades),
and `Origin` must exactly match `--remote-allow-origins` or the handshake 400s.

**REVISED SPIKE SHAPE:** the emdawnwebgpu spike must (i) **assert the adapter is what it claims** before reporting any
number, and (ii) use the handle-and-copy buffer contract from the start.

**DX ROUTE FOUND 2026-08-14 — the tracer. Corrects my earlier "no route short of a compiler".**
For ARBITRARY code that remains true, and the reason is now precise: Clang/AdaptiveCpp emit LLVM IR with PHYSICAL
pointers while Tint's SPIR-V reader assumes Vulkan LOGICAL pointers (no `VariablePointers`) — that legalisation is the
8-20 person-months. Reflection is NOT a route either: C++26 P2996 reflects declarations and members, never function
bodies. **But GR4 already imposes the style that makes tracing work:** `Block.hpp:1780,2022` instantiates a
type-generic `processOne` with `stdx::simd<T>` when `can_processOne_simd` holds. Instantiate that SAME template with a
symbolic sample type, record the arithmetic DAG at runtime, print WGSL/GLSL — the C++ compiler does the parsing, no
frontend is written. One source, four executions: CPU scalar, `std::simd`, SYCL bit-copy, WebGPU traced. Author types
nothing new. **Limits (honest, and they nearly coincide with `std::simd`'s own):** data-dependent `if`/`for` untraceable
→ `gr::select` mirroring simd `where()`; cmath via ADL-visible `gr::math`; v1 bakes settings as constants (re-trace +
cached pipeline per settings epoch), param-slot lowering via a reflected shadow block is the upgrade. **Cost ~1-2 kSLOC,
weeks not person-months.** Precedent that runtime-built IR→WGSL is maintainable: Halide's WebGPU/WGSL backend (PR 6492).
Covers exactly the brief §19 corpus 1-5 (gain/add/clamp/polynomial/complex-mult) = the auto-parallel tier.
**Three-lane browser story, and the PoC supplies the first lane:** making BLOCK IDENTITY the unit of shader reuse means
the stock block library ships pre-authored fragments, so users composing stock blocks get browser GPU with ZERO
authoring · custom traceable `processOne` → tracer · everything else → WASM CPU with simd128. Only custom BULK bodies on
the browser GPU stay compiler-gated — document that, do not promise it. Note SYCL itself never reaches the browser
(AdaptiveCpp has no WASM device backend), so goal 1 holds fully native, mostly-not-fully in a browser.

**GLSL + WebGPU COEXIST (recommendation 2026-08-14), with one contract.** Keep GLSL as (i) the native thin backend for
"only OpenGL installed" and (ii) the fragment proving ground feeding `GLSL2WGSL`. Population for (i) is real but small
and shrinking (Linux VMs/containers with GL-but-no-Vulkan, old iGPUs; macOS never served — GL caps at 4.1, no 4.3
compute), and lavapipe gives Vulkan almost anywhere llvmpipe gives GL; wgpu-native ships a GL/GLES backend that may
subsume it (UNVERIFIED: needs GLES 3.1 for compute on a strictly-GL-4.3 box). Weigh against a large binary dep vs the
314-line zero-dep `GlComputeContext` with its surfaceless-EGL CI lane.
**LANDMINE: `GLSL2WGSL.hpp:87-107` passes unrecognised lines through SILENTLY**, so a fragment can work on GL and emit
broken WGSL undetected. Freeze the `ShaderFragment` GLSL subset as a contract with **per-fragment GLSL→WGSL round-trip
tests in CI**. It only ever covered the element-wise subset — `GlslFFT`'s Stockham staging does NOT translate (vec2,
binding 2, format-baked constants); its STRUCTURE ports, its text does not, so REGENERATE it per backend, never
translate. Retire the GL dispatch path only after a native `webgpu.h` backend is proven on an actual GL-only machine.

**MEASURED 2026-08-15 — a ~2.2 ms FIXED round-trip floor per WebGPU dispatch** (Asyncify + Dawn + CDP), dominating 9 of
12 benchmark cells. It is a LATENCY floor, not bandwidth, so it does not amortise until the quantum is large: measured,
work needs roughly `N*batch >= 128k samples` to escape it. Floor-corrected SwiftShader compute is ~1.0-1.5 GFLOP/s. The
genuinely good news: the mandatory WASM-heap<->GPUBuffer copy adds only **4-16%** on top of dispatch, NOT multiples —
so the structural browser copy is not catastrophic (contrast the native 38 ns/sample USM read-back bottleneck).
**MAINTAINER DECISIONS 2026-08-15:** suspension via **Asyncify** (works in every browser today; the whole WASM build
pays the instrumentation cost, accepted) · quantum: **set a larger minimum BUT dispatch regardless** — recommend the
large quantum through the port's minimum/recommended buffer size and warn once when below it, do NOT refuse ·
scope: **full** — a real `DeviceContextWebGpu` + the FFT dispatching to it, run standalone AND in a Graph.
`712e8534` already landed a WGSL Stockham FFT validated in-browser against `gr::algorithm::FFT` (rel err ~1.6e-07 flat
across a 64x range in N), so the kernel is known-correct before integration.

**CONSTRAINT 2026-08-14 (maintainer): WebGPU must be the ONE backend that safely falls back to CPU on WASM when it is
unavailable.** Not a nicety — WebGPU is flag-gated in Chrome on Linux and not default in Firefox on Linux, so on the
dev/CI platform the fallback is the COMMON path, not the exception. A WASM build that hard-fails without an adapter is
useless there.
The mechanism already exists and needs no new concept: `registerGlslRuntime()` publishes `gpu:glsl` only when a context
can actually be created, `DeviceContextRegistry::tryResolve` returns nullptr otherwise, and `dispatchCpuFallback` runs
the block on the CPU warning once with the reason (`ExecutionStrategy.hpp:153-154`). `registerWebGpuRuntime()` must
follow exactly that: **no adapter -> do not publish `gpu:webgpu`**, and every block silently lands on the WASM CPU path
(simd128), which is the third lane of the browser story anyway.
**THE ONE GENUINE WRINKLE — async availability vs synchronous registration.** `navigator.gpu.requestAdapter()` is a
PROMISE. You cannot synchronously answer "is WebGPU available?" at static-init time on the browser main thread, which is
where runtime registration happens today. Three candidate resolutions, DECIDE BEFORE writing the context:
**DECIDED (maintainer, 2026-08-14): demote-on-dispatch-failure as the default, PLUS an early non-blocking query.**
`registerWebGpuRuntime()` kicks off `requestAdapter()` and does NOT await it; `gpu:webgpu` is published optimistically so
nothing in the startup contract changes. If the promise resolves with no adapter BEFORE the graph runs, the domain is
withdrawn there and then — the early heads-up — and every later dispatch takes the CPU path with no failed call at all.
If a dispatch happens first, it must not block: it fails, demotes the domain, and warns exactly as
`dispatchCpuFallback` does. Never await a promise that may never resolve.
**TWO CONSEQUENCES THIS IMPLIES, neither obvious, both to handle when the context is written:**

1.  **`DeviceContextRegistry` has no withdrawal.** It offers `registerContext`/`tryResolve` only — there is no way to
    remove or disable a domain once published. Demotion needs one (unregister, or a served/disabled flag that
    `tryResolve` honours). Small, but it is new registry API, so decide its shape rather than bolting it on.
2.  **The per-block scheduler cache would outlive the demotion.** `cb083610` caches the resolved
    the resolved device context in the block after the first successful resolve (reset only when settings change,
    `Block.hpp` device-scheduler reset on `applyStagedParameters`). A block that resolved BEFORE demotion keeps
    dispatching into a dead context. Demotion must invalidate that cache for every block, or the withdrawal is
    cosmetic — this is the failure mode that would look like a crash long after the adapter question was settled.

**MAINTAINER DECISIONS 2026-08-14:** edge-identity done at **FULL scope** now (edge-level predicate + edges carrying a
`DeviceBuffer`, not just GLSL pooling). WebGPU target is **BROWSER-ONLY** for this branch — `GR_ENABLE_WEBGPU` gates an
emdawnwebgpu-only context, **no native dependency lands**, and the Dawn-vs-wgpu-native choice is deferred until someone
needs native. Note `emdawnwebgpu` IS the Dawn team's Emscripten port, so the browser path already runs Dawn; only the
native link target was ever open. wgpu-native has prebuilts (9.3 MB Linux `.so`) but its README says it does not yet
implement the stable `webgpu.h`; Dawn leads the header and builds via plain CMake (`DAWN_FETCH_DEPENDENCIES=ON`).

**SCOPED: EDGE IDENTITY, NOT POINTER IDENTITY — do this BEFORE the tracer.** `isDeviceAccessible(const void*)`
(`DeviceContext.hpp:73`, overridden `DeviceContextSycl.hpp:67`) and its four call sites
(`ExecutionStrategy.hpp:408,409,501,502`) are pointer-shaped, so `Residency::opaque` can NEVER pass them — GLSL SSBOs
today, WebGPU `GPUBuffer` tomorrow. Consequence today: `dispatchGlsl` allocates and frees both edge buffers on EVERY
work() call and round-trips the data (`ExecutionStrategy.hpp:272-284`, its own TODO). Left unfixed, WebGPU lands with
the ~100x host-boundary factor already conceded.
**Shape:** replace the pointer predicate with an edge-level question — "can this context execute against THIS edge
buffer in place?" — answered from the edge's `DeviceBuffer`/domain rather than from an address. Keep
`isDeviceAccessible` for the USM/CUDA-VMM raw-pointer edges that legitimately have one. Then pool the GLSL edge buffers
across calls instead of per-call allocate/free. ~80-120 lines across DeviceContext + ExecutionStrategy + the two
contexts; folds into the residency-by-edge-domain work already planned. **Prerequisite for any WebGPU perf claim.**

**WebGPU AS A BACKEND — CI-TESTABLE, MEASURED THIS SESSION.** The PoC built clean for WASM and ran correct compute in
headless Chrome (bit-exact vs CPU reference) and Firefox (1.19e-7). So a CI lane is achievable TODAY:
emsdk >= 4.0.10 + `--use-port=emdawnwebgpu`, serve over http (file:// will not load WASM), drive headless Chrome
(`--headless=new --enable-unsafe-webgpu`), poll results via CDP `Runtime.evaluate` after a REAL wall-clock wait —
`--dump-dom --virtual-time-budget` truncates because virtual time does not wait on async GPU IPC. Firefox 153 headless
speaks WebDriver BiDi not CDP: no `/json`, handshake targets `/session`, `Origin` must match `--remote-allow-origins`.
**What CI can and cannot prove:** correctness YES; performance NO — headless Chrome self-selects SwiftShader even with
`--enable-features=Vulkan` (ANGLE's backend choice and Dawn's are independent), so **assert the adapter string on every
run** or every number is a software number. Linux browser WebGPU is still flag-gated/not-default, so CI cannot assume a
hardware adapter at all.

**WebGPU SUBSTRATE DECIDED 2026-08-13: emdawnwebgpu + the standard `webgpu.h`. Canonical, because the alternative is
already deleted upstream.** `-sUSE_WEBGPU` / `<emscripten/html5_webgpu.h>` was deprecated in emscripten 4.0.10
(2025-06-07) and **REMOVED in 4.0.18** (2025-10-24); Emscripten's own bindings were declared unmaintained (issue #24265).
`--use-port=emdawnwebgpu` (needs emsdk >= 4.0.10) is therefore the only supported Emscripten path. **The PoC's
hand-written `EM_ASYNC_JS` glue is the backport, not the port** — it shares zero code with any native target and is a
dead end beyond its proof value. Keep it only as a parity reference.

- **ONE C++ CODE PATH for browser and native.** A `DeviceContextWebGpu` written against `webgpu.h` compiles unchanged
  for WASM (emdawnwebgpu) and native (Dawn from CMake, or **wgpu-native prebuilts — verified: v29.0.1.1, Linux .so
  9.3 MB, ships `include/webgpu/webgpu.h`**). NOT shared: instance/adapter bootstrap and completion waits — native may
  block in `wgpuInstanceWaitAny`, the browser must yield to the event loop (JSPI/Asyncify or a callback pump). That is a
  synchrony seam of tens of lines behind one `#ifdef __EMSCRIPTEN__`, **not a fork**.
- **Caveats to carry, not rediscover:** `webgpu.h` has **no releases and no v1.0** — rolling `main`, two breaking waves
  already landed (futures-based async 2024-07, `WGPUStringView` 2024-09); core compute stable since, and the port
  absorbs the churn. **On Linux, WebGPU is still flag-gated in Chrome and not yet default in Firefox** — exactly the dev
  and CI platform. Portable compute baseline is **f32 only**: no timestamp queries (Chrome-only), no subgroups
  (Chrome 134+), no assumed f16. Spec-floor limits: 128 MiB max storage-buffer binding, 256 MiB max buffer.

**THE BUFFER CONTRACT — SETTLED. The view carries a POINTER; the framework carries a HANDLE; handles stop at the
kernel-launch boundary.** Dispatch resolves `devicePointer<T>()` once on the host and the SYCL lambda captures a raw
`T*` exactly as today, so `InputViewLike`/`OutputViewLike` stay pointer+size shaped and **never see a handle type**.
Opaque backends never construct a view at all — they are gated out of the kernel-body tiers and reached only through the
expert hatches, which take the context plus host spans and copy through handles.

```cpp
enum class Residency : std::uint8_t { invalid, host, shared, devicePtr, opaque };
struct DeviceBuffer {                       // 24-byte trivially-copyable POD, replaces every void* in the virtual surface
    std::uintptr_t token = 0; std::size_t bytes = 0; Residency residency = Residency::invalid;
    explicit operator bool() const noexcept;
    template<typename T> T* devicePointer() const noexcept;   // null unless a kernel may index it
    template<typename T> T* hostPointer()   const noexcept;
};
// replaces allocate{Device,Host,Shared}Raw / deallocateRaw / copy{HostToDevice,DeviceToHost}:
virtual DeviceBuffer allocate(std::size_t bytes, std::size_t align, Residency wanted) = 0; // invalid = "cannot serve", never a lying token
virtual void deallocate(DeviceBuffer) = 0;                    // residency picks the path -- deletes the 64->32-bit id-truncation heuristic
virtual void upload(const void* host, DeviceBuffer dst, std::size_t bytes) = 0;
virtual void download(DeviceBuffer src, void* host, std::size_t bytes) = 0;
virtual void copyDeviceToDevice(DeviceBuffer src, DeviceBuffer dst, std::size_t bytes) = 0;
```

**Why the live GLSL bug becomes UNREPRESENTABLE:** `dispatchDeviceBulk` requests `Residency::shared` scratch; GLSL has no
shared memory so it returns `invalid`, and the kernel-body tiers fall back **naming the reason** instead of host-writing
through an SSBO id (`ExecutionStrategy.hpp:433`, `ParallelFor.hpp:26-28`). There is no type-level path from an opaque
token to a `T*`. SYCL is untouched: shared USM yields `devicePointer<T>()`, edge elision keeps using
`isDeviceAccessible(span.data())`, `processBulk_sycl` never sees any of this. Zero CPU tax — all of it lives in
`device/` behind `GR_DEVICE_HAS_ANY_BACKEND`.
**Named friction, do NOT block the contract on it:** device-resident _edges_ hand out raw `span.data()`, fine for
USM/CUDA-VMM, impossible for WebGPU. v1 keeps WebGPU edges host-resident (the copy is mandatory anyway); device-resident
WebGPU edges later mean the edge carries a `DeviceBuffer` beside the ring — folds into residency-by-edge-domain.
**Size:** ~300 lines touched, net +130, across DeviceContext, three backends, ExecutionStrategy, `GlslFFT._twiddleSSBO`,
and ~50 call sites in 17 files. **Breaks:** out-of-tree `DeviceContext` subclasses and anything holding a `T*` from
`allocateDevice<T>()`.

**ADOPT 1 — `InputViewLike`/`OutputViewLike` (maintainer's own candidate).** A kernel-facing concept with no tags and no
consume/publish, distinct from `ReaderSpanLike`/`WriterSpanLike`. Pays twice: it deletes the poison-member hack in
`DeviceSpans.hpp` (members that satisfy the span concepts then `static_assert` on any call) AND the **runtime**
`_tagWasDropped` canary plus its per-dispatch allocate/write/check, because a view-constrained `processBulk` cannot name
`publishTag`. Shape reuses the existing `ConstSpanLike`/`SpanLike`; re-express the span concepts as REFINEMENTS so
subsumption orders overloads — then a host `InputSpan` satisfies both and **one `processBulk` serves host and device**.
Tier selection stops being structural inference and becomes a block-author declaration. Net ≈ −65 lines.
COST: a block written `const processBulk(InputSpanLike auto&)` stops matching the device gate and falls back to CPU
until re-constrained (in-tree: `qa_DeviceBlockStyles.cpp:102`); downstream blocks of that shape UNVERIFIED.

**ADOPT 2 — typed device buffer handle, replacing `void*` in `DeviceContext`'s virtual surface.** Not WebGPU groundwork:
a **live defect**. On a GLSL context `allocateShared` returns an SSBO _token_, yet `ExecutionStrategy` host-writes
through it (`*dDropped = 0U`) and `ParallelFor` host-loops for non-SYCL backends — the framework bulk/auto-parallel tiers
dereference fake pointers (reachable crash; whether a test hits it UNVERIFIED). `deallocateRaw` also disambiguates
token-from-heap by truncating 64-bit pointers to 32-bit GL ids — latent wrong-`glDeleteBuffers`. ≈100-150 lines.
**Both 1 and 2 must precede PR-B, which would otherwise freeze the wrong block-author contract and the `void*` ABI.**

**ADOPT 3/4 (not urgent).** GLSL constants as _uniforms_ rather than baked `#define`s — `ShaderFragment` bakes values
into source, so every settings change recompiles and caches forever; the header already made `uCount` a uniform for
exactly this reason (~70 lines, no ABI impact). And a minimal `DeviceCapabilities` (2-4 fields **with readers today**)
to replace the hard-coded `requires std::same_as<T, float>` gates.

**REJECTED, with reasons — do not re-propose without new evidence:**

- **async / pending dispatch result** — the measured bottleneck is the host-boundary transfer strategy (~124x from
  staged pinned copies), NOT dispatch synchrony; the seam is header-internal so a later refactor is not a user ABI
  break. (This corrects my own earlier advice to settle the async contract before the re-cut.)
- capability-MATCHING dispatch — `compute_domain` is the user's explicit placement, not an inference to replace.
- versioned pipeline/shader cache keys — the cache is per-context and in-process; version keys need a persistent cache.
- `DeviceExecutionPlan` / graph flattening / fusion — no consumer; the brief itself defers fusion.
- Level-2 op registry — `SyclFFT`/`GlslFFT` already serve this informally; a registry = Interface/Impl with one impl.
- `gr::algo` facade, adapter software/hardware policy, `Backend::WebGpu` enum — no browser backend in tree.
- the brief's L0-L3 tier taxonomy — GR4's five tiers are finer and documented; at most a doc cross-reference.

### 🎯 ONE FFT BLOCK — maintainer decisions 2026-08-10, and the ordered plan (phases 1-3 done, rest OPEN)

**End state: a single `gr::blocks::fft::FFT` that does the old DataSet spectrum work AND the new device work.**
`fft.hpp`'s block and the `FFT2` name both disappear; `qa_FFT2Parity` is transitional scaffolding.

**Decisions (settled — do not relitigate):**

1. **Bin layout — fix it, do not preserve the quirk.** Real input → **N/2+1 bins, DC..Nyquist inclusive** (standard rfft
   convention). Complex input → **full N bins, fftshifted** so negative frequencies come first. **All four signals
   (magnitude, phase, Re, Im) share ONE layout and align with the frequency axis.** Today they do not: magnitude/phase
   use `[0,N/2)` (has DC, drops Nyquist) while Re/Im use `last(N)` = `[1,N/2]` (drops DC, keeps Nyquist), and for
   complex input magnitude/phase are fftshifted but Re/Im are not. Both blocks share the quirk, which is _why_ the
   12 288-assert parity sweep is green — bit-identity proved faithful reproduction, not correctness.
2. **Sequencing:** apply the layout fix to **both blocks in one commit** and update the golden expectations with it, so
   parity stays green throughout and keeps working as the retirement gate. `fft.hpp` is no longer off-limits.
3. **Name:** the unified block takes **`FFT`**; register the `T` set FFT registers today (float, double) plus the
   complex ones, so existing flowgraphs and the registry keep resolving `FFT`. No `FFT2` alias.
4. **Coverage:** before deleting `qa_FFT2Parity`, **capture its golden DataSets (post layout fix) as committed reference
   vectors** and convert the test into a self-contained regression against them. Otherwise 12 288 assertions of
   cross-window / cross-flag coverage vanish with `fft.hpp`.

**DEVICE UNWRAP IS POSSIBLE — the reported blocker was a red herring.** `std::llrint` failing to resolve under
AdaptiveCpp SSCP (`__acpp_sscp_llrint_f32`) is a library-availability problem; the rounding is not needed at all.
`atan2` yields (-pi, pi], so consecutive differences lie strictly in (-2pi, 2pi) and the per-element correction is
bounded to {-1, 0, +1}: `c = (d > pi) ? -1 : (d < -pi) ? +1 : 0` — pure comparisons, device-safe, and _exactly_ the
strict `|d| > pi` convention including the +/-pi tie that `llrint` was chosen for. The integer prefix sum stays exact.
**Precondition: unwrap must run on radians before any degree conversion** — the (-2pi, 2pi) bound breaks otherwise.
This is what removes the GPU->CPU->GPU round trip; the maintainer flagged it as performance critical. Verify on hardware,
do not just assert it.

**Ordered plan (each step independently gated):**
`A` device-safe unwrap (comparison form), verified in a real kernel — unblocks keeping magnitude/phase on device ·
`B` bin-layout fix in both blocks + golden expectations, one commit · `C` capture reference vectors ·
`D` rename to `FFT`, delete the old block, registry entries for both type parameters (the single-`[T]`
`GR_REGISTER_BLOCK` form does not cover a two-parameter registration — that is the open mechanical problem) ·
`E` delete `qa_FFT2Parity` once `C` has landed · `F` the PR re-cut.

**Phases 1-3 state (uncommitted, 6 files, +725/-31, all verified by me, NOT reviewed — the Fable review hit a session
limit):** `qa_FFT2Parity` 12 288/421 · `qa_FFT2` 2 423/48 · `qa_fourier` 1 897/18 unchanged ·
`qa_algorithm_fourier` 742/37 · `qa_SimdFFT` 25 017/237. Residual gaps: spectrum mode has **no registry entry**;
device spectrum path is SYCL + `complex<float>` only (`SyclFFT` is hardcoded to it — pre-existing), R2C and double stay
host, GLSL has no spectrum path; the output port deliberately omits `RequiredSamples<1,1>` so several DataSets can leave
one call (needed for batch parallelism on device).

### 🔬 BRANCH vs. THE PREMISE — full re-analysis (2026-08-09, 3 evidence probes + 3 Fable passes, claims spot-checked)

**The premise under test:** the _same_ user-written `processOne`/`processBulk` runs on the CPU as before **and** dispatches
to a GPU. Verdict: **fully true for exactly one of five tiers**, explicit-and-fine for two, and **silently wrong for one**.

| tier                                          | user code unchanged?        | behaviour identical?                                                        | what breaks                                                                                                                    |
| --------------------------------------------- | --------------------------- | --------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| 1 expert hatch (`processBulk_sycl`/`_glsl`)   | no — new per-backend method | yes: host thread, real spans, real block; tags/state/variable-rate all work | nothing in contract; it does not _test_ the premise                                                                            |
| 1b shader fragment (GLSL)                     | no — new `shaderFragment()` | yes for element-wise float math                                             | no tag concept at all; float-only; shader cache never evicts                                                                   |
| 2 framework device-bulk (`const processBulk`) | **yes**                     | **NO, silently**                                                            | see below — three wrong-answer holes                                                                                           |
| 3 auto-parallel (`const noexcept processOne`) | **yes**                     | **yes** — the premise holds here                                            | nothing lost (tag/state access is a compile error on the CPU too)                                                              |
| 4 CPU fallback                                | yes                         | mostly                                                                      | hard-errors on span shapes a plain CPU run would serve; warn-once flag shared with tier 2 so a real fallback can be suppressed |

**Why tier 3 is genuinely sound (stronger than it first looks):** `mergedInputTag()` (`Block.hpp:1439`) and `publishTag()`
(`Block.hpp:1415`) are **non-const**, so a `const noexcept processOne` could not touch tags on the CPU either. The tier is
tag-symmetric _by construction_, not by luck.

#### Tags on device

- **Works:** tier-1 hatches (full read/publish, the branch's only device tag test, `qa_DeviceSeam.cpp:64`); `gr:`-prefixed
  and auto-forwarded tags, forwarded on the host _before_ dispatch so they survive every tier; the **tag axis of every edge
  is forced to `Access::Shared`** even on a DeviceOnly data edge (`Graph.hpp:748`, guarded by `qa_DeviceResidency.cpp:116`);
  kernel-side `publishTag` fails **loudly** (drop flag → `work::Status::ERROR`, `ExecutionStrategy.hpp:383`).
- **Silently fails:** tier-2 tag _reads_. `DeviceInputSpan::rawTags()/tags()` `return {}` unconditionally
  (`DeviceSpans.hpp:39-41`, comment `// tags never reach a kernel`). It is a deliberate decision **documented only in the
  framework's own source** — a tag-reading `const processBulk` compiles, dispatches, and computes a different answer with
  no compile-time or runtime signal. The asymmetry is the smell: **input silent, output hard error.**
  `consume()`/`consumeTags()` are likewise no-ops (`:36-37`) and the framework forces `processedIn = processedOut = count`
  (`Block.hpp:1946-1948`), so **decimators and variable-rate blocks drift**.
- **UB:** `_mergedInputTag` is a _public_ member (`Block.hpp:801`), so a `const processOne` can read it directly, bypassing
  the non-const accessor. The mirror carries an epoch-stale copy whose view aliases a host tag-ring chunk that is
  reclaimable after the work call. Nothing exercises it; **making that state non-public closes it for free.**
- **The fix, in order.** (a) _Refuse honestly, ~10 lines:_ delete `rawTags()/tags()` from `DeviceInputSpan` so a
  tag-reading `processBulk` is a compile error on the device path. (b) _Actually deliver:_ the tag axis is already
  host-accessible USM and `BasicTag<false>` is trivially copyable — stage the work quantum's `(index, view)` list into a
  USM array pre-launch and hand the span a real `(Tag*, size)`; for output give `publishTag` a claim-slot API into a
  pre-sized USM slab (atomic cursor, claim-or-drop, drop counter) drained after `.wait()`.
  **The `DeviceLogger` USM slab is exactly the right precedent** (`DeviceLog.hpp:38-48` + host drain at `:364`) — the
  output side is close to a transplant, the input side is simpler still (a plain pre-launch copy, no atomics).

#### Settings on device

The contract, precisely: **every mutation must go through the settings system; nothing enforces that.** When it does,
`_settingsEpoch` is bumped in exactly two places (`Block.hpp:960` `init()`, `:1340` `applyChangedSettings()`);
`migrateFieldsToDeviceResource()` re-seats every `PmrMigratable` reflected member onto the device resource (`:990-1003`);
the mirror is a whole-object `memcpy` of `sizeof(TBlock)` into USM (`DeviceRelocatable.hpp:106`) — **pmr members are not
deep-copied, mirror and host share backing storage** — refreshed only on epoch mismatch (`ExecutionStrategy.hpp:283`),
read-only on the device, **never copied back**. Mid-stream changes land before dispatch in the same `work()` call and the
SYCL submit blocks, so **no stale-read window exists _while dispatch is synchronous_** — that is a re-verification trigger
if dispatch ever goes async. `settingsChanged()` only ever runs on the host object.
Failure modes: (1) a **direct** pmr-member reassignment after first dispatch is a device use-after-free (no epoch bump, old
storage freed, mirror keeps the stale pointer) — no diagnostic, no test; (2) device-side mutation is discarded by design and
the only canary is **Debug-only, tier-3-only, once-per-epoch**, and it probes by running one un-rolled-back `processOne` on
the live block — tier 2 has none in any build; (3) `std::span` and pointer-carrying trivially-copyable members pass the
`DeviceRelocatable` gate and smuggle host addresses into kernels; (4) GLSL settings are a separate hand-authored float-only
`ShaderConst` channel with an insert-only program cache.

#### Achieved

- tier-3 premise holds byte-identically and output-identically, with tag/state symmetry proven by the non-const barrier
- one-seam composition: a single `if constexpr` in `Block<T>` (`:1915-1958`); the CPU path below it is untouched
- settings coherence is sound for everything routed through the settings system (epoch + synchronous mirror + pmr re-seat)
- expert hatches work fully, three production users, and honour their own consume/publish via `blockManagedIO` (`:1930`)
- loud failure on kernel-side tag publish and on unresolvable backends — no silent backend substitution
- tag axis host-accessible on every edge including DeviceOnly, regression-tested
- `DeviceRelocatable` compile-time member gate that names the offending member
- one `Block<T>` layout in every configuration (ODR/plugin safety), device code otherwise compiling out to a warn-once
- the one measured win is real: 1.45-1.47x on a device-only edge, backed by load-bearing CUDA-VMM + capability registry
- **goal 6 flips PARTIAL→MET once this working tree is committed** (GL lane, GL under acpp's `-Werror`, SYCL-CPU fallback)

#### Missing

- `[breaks-the-premise]` tier-2 **tag reads silently empty** (`DeviceSpans.hpp:39`), untested
- `[breaks-the-premise]` **silent perimeter**: an ineligible block with a device `compute_domain` is a complete no-op (the
  warning lives _inside_ the `if constexpr`), and a typo'd domain (`"cuda"`, `"gpu-sycl"`) parses to `host()` silently
  (`ComputeDomain.hpp:79-81`) — ~15 lines to fix both
- `[breaks-the-premise]` tier-2 **forced full consumption** (`Block.hpp:1946`); the comment claiming span-equivalence at
  `ExecutionStrategy.hpp:311` overclaims
- `[breaks-the-premise]` **F1**: device-side mutation is not a compile error anywhere and the canary is Debug/tier-3 only
- `[limits-the-premise]` direct pmr reassignment after first dispatch = device UAF; no guard, no test
- `[limits-the-premise]` `std::span`/pointer-carrying members pass the relocatable gate
- `[limits-the-premise]` the documented per-member fallback diagnostic (`ExecutionStrategy.hpp:400-404`) is **dead code** —
  its only call site is pre-gated on `DeviceRelocatable`; `USER_API_GPU_Blocks.md:90-97` documents an unreachable message
- `[limits-the-premise]` device tag/settings coverage is **one test on one tier**; no mid-run settings change under the
  scheduler — below the CLAUDE.md §7 bar
- `[polish]` `_deviceFallbackWarned` conflates tier-2's serial notice with the genuine CPU-fallback warning
- `[polish]` `Block.hpp:811` says "32 bytes"; actual unconditional growth is **40** (8+24+8) — and it is an ABI break vs
  main in every configuration, worth calling out at merge

#### Minimalism verdict

KEEP: `ExecutionStrategy` (521, −30 dead), `DeviceRelocatable` (126), `ComputeDomain` (198), the context/runtime/USM/span
plumbing (~1,400), CUDA-VMM + `MemoryResourceCapabilities` (377, backs the only measured win), GL backend (~340, mandated).
SPLIT: `DeviceLog` (~550 — own PR, and it is the
tag-staging precedent so keep it alive). **`FFT2`/`FFT`: the ~624-line "duplication" is REFUTED (measured 2026-08-09) — do not re-propose it.** They are
different blocks: `FFT` is `Resampling<1024,1>` emitting a `DataSet` (window, magnitude/dB, phase/unwrap, R2C real input,
signal metadata); `FFT2` is 1:1 complex→complex with `inverse`, batching and device dispatch. Neither is a superset, so
"identical or better" is not evaluable. And they **already share one engine**: `FFT`'s default `FourierAlgorithm` and
`FFT2::_cpuFft` are both `gr::algorithm::FFT`, whose C2C path reinterprets the aligned complex buffer and calls the very
same `SimdFFT::transform` — output is **bit-identical** (0.0 abs and rel error at N=64/1k/4k/64k, float and double) and
throughput is within noise (ratio 0.97–1.04). Everything that differs sits _above_ the shared engine, so nothing is
removable by delegating harder. FFT2's conj-forward-conj-scale inverse round-trips at pure round-off. The only real gap
is device dispatch for `FFT`, and that needs **new** device kernels for windowing/magnitude/dB — a feature, not a
refactor. Two bonus findings: `gr::algorithm::FFT` hardcodes `Order::Ordered` on every path, so **neither block can reach
SimdFFT's unordered mode, which measured 12–26% faster**; and `FFT`'s single-argument `compute(_inData)` allocates a
fresh output vector per call, which `FFT2`'s two-argument form avoids. **KEEP (maintainer decision, 2026-08-09, supersedes the DELETE verdict): the whole GLSL family** — `ShaderFusion.hpp`
(156) and `GLSL2WGSL.hpp` (115). Being test-only today is not the point: they are the seed of the WASM/WebGPU path, where
fusing an element-wise chain into a single compute pass is not an optimisation but the difference between viable and not.
They need further work, not deletion. (`ShaderFusion.hpp` was deleted earlier in this batch and has been restored;
`qa_DeviceContext` is back to 24 asserts in 5 tests.) FOLD: `blocks/device/` (89, a directory containing only a test).
**Net ≈ −2,050 production lines with zero measured capability lost.**

#### ✅ ITEMS 1-6 IMPLEMENTED 2026-08-09 (committed) — what actually shipped, and what the review changed

1. **Tier-2 silent no-ops poisoned.** `DeviceInputSpan::{rawTags,tags,consume,consumeTags}` and
   `Device{Output,TagWriter}Span::publish` are now member templates whose body `static_assert`s. The _declarations_ stay,
   so `ReaderSpanLike`/`WriterSpanLike` are still satisfied and the spans stay trivially copyable; only a real call is a
   compile error. Proven with a scratch TU: baseline compiles with both concepts satisfied, each call fails with its
   message. `publishTag` deliberately keeps its runtime drop-flag (a data-dependent publisher may never fire).
2. **Perimeter.** A device `compute_domain` with no available path warns once (the old warning lived _inside_ the
   `if constexpr`); an unrecognised kind warns once too — keyed on a ':' in the string, exempting the grammatical
   `host:...`. Residual: colon-less typos (`gpus`, `sicl`) still parse to host silently.
3. **Tier-2 overclaim removed** from the doc comment; the serial notice now names the forced sample count.
4. **Canary in every build, on a bit-copy** (`mutatesItsOwnState`), extended to the bulk tier. **Two hazards the Fable
   review caught and that are now fixed:** the first draft probed bulk with `_size = 1` into a stack scratch — a
   fixed-chunk block writing its contractual N samples would have smashed the stack _in release_; it now hands over the
   real spans (right size, and the probe's output is overwritten by the dispatch that follows) and runs only at the host
   boundary, never on a device-resident edge. And the bit-copy aliases the original's pmr storage, so a probed call that
   reallocated a `mutable` pmr member would free the _live_ block's memory — hence the new `DeviceProbeSafe` concept:
   only blocks whose own members are all trivially copyable are probed. Also: the probe no longer dereferences
   `inSpan[0]` (device-only memory on some edges), and the shadow-less latch is an `atomic_flag`, not a plain bool.
5. **Non-owning views** (`std::span`, `std::string_view`) rejected by the relocatable gate — a denylist, so `std::array`
   stays eligible. `firstStaleMirrorMember` names a pmr member reassigned outside the settings system (Debug-only).
6. **Diagnostics reachable**: the offender name moved to the generic else-branch in `dispatch()`; `_deviceFallbackWarned`
   split per cause so the serial notice no longer swallows a genuine fallback warning; the `32 bytes` comment is now 40
   and says it is an ABI break.
   **Verified:** gcc-15+GL, AdaptiveCpp (SYCL+GL), and CPU-only GL-OFF all build clean under `-Werror`; the whole acpp
   device suite passes with the GPU visible and hidden, unchanged assert counts; `qa_DeviceRelocatable` 7/4 → 11/6.
   **Still open from item 4:** a block owning pmr members is not probed at all (`DeviceProbeSafe` excludes it), so F1 is
   narrowed, not closed. A true compile-time refusal needs C++26 reflection.

#### Smallest set of changes that would make the premise true (<150 production lines, items 1-6)

1. poison tier-2 tag reads (~10) · 2. warn on the silent perimeter + reject unknown domain kinds (~15) ·
2. stop overclaiming tier 2: fix the comment, refuse blocks whose `processBulk` calls `consume()` (~10-20) ·
3. F1: extend the mutability canary to tier 2, run it in **all** builds on first epoch use, restore the snapshot (~30) ·
4. reject `std::span`-like members; Debug guard comparing mirror-time `(data,size)` against the host fields (~30) ·
5. make the dead diagnostic reachable; split `_deviceFallbackWarned` per cause; fix the 32→40 comment (~10) ·
6. tests: a non-empty tag stream through tiers 2/3/GLSL asserting the refusals, one mid-run settings change (~150 test) ·
7. commit this working tree — goal 6 is not banked until then · 9. re-cut the PRs per the minimalism verdict.

**The honest one-sentence premise after items 1-6:** _a pure `const noexcept processOne` with scalar or pmr settings runs
on the GPU byte-identical and output-identical; every other route is either explicit per-backend code where everything
works, or is refused loudly — nothing silently diverges._

### GOAL SCORECARD (final review, 2026-08-08) — SUPERSEDED in part by the re-analysis above; note its numbering is offset from §2 (its "4" is §2 goal 5, its "5" is §2 goal 8, its "8" scores the §1 use-cases) — use this, not §2's aspirations

`1` PARTIAL (correct, but "70-80% of device peak" is unmeasured and transport-refuted) · `2` PARTIAL (both hatches live;
"~100%" never shown in-graph) · `3` **MET** · `4` **MET** (~40 B/block traded deliberately for one-layout ODR safety) ·
`5` **MET** (one `if constexpr` seam + ~138 supporting lines) · `6` PARTIAL (**GL has no CI — follow-up 1**) ·
`8` PARTIAL (use-case A met end-to-end; **B proven for exactly
one topology** — two adjacent blocks, one interior edge, one device, CUDA-VMM opt-in, 1.45-1.47x; no chains >2, no fan-out
with mixed residency, no automatic transfer insertion, no multi-device).
**Honest restatement of goal 1, to replace §2's wording:** _a `const noexcept processOne` runs on a device unchanged and
correct; it approaches device-limited throughput only when its edges are device-resident and work quanta are large
(>=8-32 MiB); through host edges it is transport-bound at 0.1-0.8x of the host CPU._
**Re4 is months, not weeks:** browsers have **no compute shaders** (SSBO needs GL 4.3, WebGL2 is GLES 3.0), so
"GLSL production-grade for the browser" realistically means a **WebGPU/WGSL `DeviceContext`** plus a translate-GLSL vs
author-WGSL decision, an llvmpipe CI leg, and a non-float buffer policy. Re1 ~1-2 weeks (cheapest now, zero call sites);
Re3 ~1-2 weeks with the naming decision first.
**Also flagged as not carrying its weight:** DeviceLog
~550 lines (serves none of the 8 goals — split as `PR-H`) · `FFT`/`FFT2` duplication 624 lines (Re3).

### SIZE OF THE CHANGE (measured 2026-08-08)

45 commits, 86 files, **+10,217 / -151**. Production **5,288** (device layer 2,805 · core headers 826 ·
algorithm/FFT 602 · blocks 300) · tests & benchmarks **4,710** · docs 145 · build 74. Test:production ~0.89:1 — appropriate
for hardware-touching infrastructure.

### MAINTAINER DECISIONS (2026-08-08) — treat as settled

re-use and refactor. Do NOT delete it (an earlier audit recommended deletion; overruled).

- **Keep it MVP for users who do not use device compute** — CPU-only cost and complexity is the priority.
- **`FFT2` is a temporary design-phase name.** It should REPLACE `FFT` if it is functionally and performance-wise identical
  or better; otherwise reconcile them. Do not ship `FFT2` as a permanent public name.
- **GLSL must become production grade** — it is needed to run blocks on WASM/in the browser. It is a shipping requirement,
  not an experiment. (This makes the new explicit `GR_ENABLE_GL_COMPUTE` option a foundation, not a demotion. Note
  `device/GLSL2WGSL.hpp` was queued for deletion as dead — it may instead be on the critical path for WGSL/WebGPU.)
- Commit `34` (dead `BlockModel::processBulkDevice` virtual for the abandoned managed sub-Graph) and commit `43` DROPPED.
  Fable dissented on `43` (keep the out-of-place code, drop only the `perf:` claim) — recoverable from the backup branch.

### PRE-WORK: done vs outstanding

**Done** (`c93d932c` + the drops): 2 zero-reference headers deleted (`ShaderCache`, `DeviceBuffer`); **goal 4 restored** —
`GR_ENABLE_GL_COMPUTE` is an explicit option, default OFF, fails configure if packages are missing, and compile+link now
share one gate (it previously switched itself on via `__has_include(<EGL/egl.h>)`, so any box with mesa headers compiled the
GL stack and `ExecutionStrategy` into every TU including `Block.hpp`); the `HasSyclBulk = false` concept deleted and
`DeviceEligible` simplified. Gated: CPU-only gcc15 (GL OFF) and acpp with GL ON, tests green.
**Outstanding:** rename the colliding `gr::device::SchedulerRegistry` (clashes with `gr::SchedulerRegistry`,
`BlockRegistry.hpp:152`); make `ExecutionStrategy::dispatchTiers` private (its only external caller was commit 34);
delete `GLSL2WGSL.hpp` + `ShaderFusion.hpp` **only if Re4 does not need them** (both referenced solely by
`qa_DeviceContext.cpp`); `Block<T>` layout/ODR fix (device state as an unconditional opaque member so layout cannot differ
between backend-on/off TUs — a plugin hazard); full four-compiler `-Werror` matrix re-run (gcc15, clang20, acpp, emscripten).

### SCOPE AUDIT FINDINGS (verified by grep, not assumed)

~1,350 of ~6,600 production lines served no live code path. Zero-reference: `ShaderCache`, `DeviceBuffer` (both now gone).
Test-only: `GLSL2WGSL` (115), `ShaderFusion` (156). `execution/` (755) — included only by `device/SchedulerRegistry.hpp`
plus two tests, and its one production consumer is used solely as `->context()`/`->backend()`, i.e. a
`DeviceContext*` in a costume. `Block<T>` gained "one seam plus six barnacles" (+138 lines total, seam itself is one
`if constexpr`). **Five fixes (`9`,`14`,`21`,`22`,`24`) repair files from earlier PRs, so any chronological cut ships
known-broken intermediates** — PR1's SyclFFT computes the wrong transform until PR3, PR1's GL context is non-reentrant
until PR3. The earlier stacked `pr/1..pr/6` branches encoded exactly that and have been RETIRED.

### REVISED PR PLAN (thematic re-cut; supersedes the retired chronological cuts)

`PR-0` fixes to main (`35` munmap leak + `7` move-ctor `_allocResource`, both pre-existing) · `PR-A` device runtime
(trimmed commit 1, with `9`/`14`/`24` squashed in) · `PR-B` dispatch seam (`2`+`4`+`5`+`6`+`30`+scheduler-cache, traits
cleanup, docs `13`) · `PR-C` kernel bodies + device bulk (`11`+`12`, `16`+`22`, `29`, `33`-residue) · `PR-D` GLSL domain +
shader hatch (`15`,`18`) · `PR-E` residency + precedence (`17`, `26`+`28`) · `PR-F` device-only edges (`36`+`37`, `38`) —
the one PR with a measured win · `PR-G` FFT2/SyclFFT with the `1`→`21`→`32` triple converged and six perf-test rewrites
collapsed to one · `PR-H` DeviceLog standalone (550 lines, serves none of the six goals — a different feature).
Fold fixes into origins, then re-cut thematically (closer to option (c) than (b)). ~1 day with per-commit gating.

**COMMIT → PR MAPPING, re-derived 2026-08-14 against the current 56. Supersedes the 47-commit table.**
Backup ref `backup/pre-pr-recut` = `ed402137` (commit 47) — it predates 48-56, so branch a fresh backup before executing.

| PR                                                                                                                      | commits (oldest first)                                                                                                                                                          | n   |
| ----------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --- |
| `PR-0` pre-existing core fixes, cherry-pickable to main                                                                 | `3962db30` `7cd87d52`                                                                                                                                                           | 2   |
| `PR-A` device runtime + compute domain                                                                                  | `ed478f40` `4c1e2792` `c45257d6` `04cb4f4b`                                                                                                                                     | 4   |
| `PR-B` dispatch seam + the two contracts                                                                                | `2e1ab891` `f6fde29a` `e09bc1b9` `8ce71b61` `194824d6` `df636373` `cb083610` `6143c36c` `5115c1e8` `ed402137` `6e4fb08d` `ef6f2a16` `27b1f48c`                                  | 13  |
| `PR-C` kernel bodies + device bulk                                                                                      | `7938b249` `77797ed7` `6be296d4` `cd84a67c` `3a2b6645` `4d56a610`                                                                                                               | 6   |
| `PR-D` GLSL domain + shader hatch                                                                                       | `fb44a5b1` `39cc4e3d` `c0cff229` `589b7b40` `c93d932c`                                                                                                                          | 5   |
| `PR-E` residency + resource precedence                                                                                  | `a0d4513c` `9d4d5bff` `929fbd1b`                                                                                                                                                | 3   |
| `PR-F` device-only edges (the one measured win)                                                                         | `b8c834f5` `35865fc7` `8b2dedf0`                                                                                                                                                | 3   |
| `PR-G` FFT: SYCL/GLSL engines, the unified block, its tests                                                             | `4c1f51b2` `17939e48` `15ede959` `9b43ef3c` `10da8fc2` `8a20b540` `6a5fed4b` `b5b8d66b` `536f125f` `9b6a9552` `351bc88f` `92b098d6` `8778c2f3` `a60560a5` `fd72eb9f` `076d21eb` | 16  |
| `PR-H` DeviceLog standalone                                                                                             | `28b70d3d`                                                                                                                                                                      | 1   |
| `PR-I` CI lanes + device test hygiene                                                                                   | `d2eaa092` `d1d33bf1` `ef551b8f`                                                                                                                                                | 3   |
| All 56 accounted for (2+4+13+6+5+3+3+16+1+3).                                                                           |
| **What changed from the 47-commit table:** `PR-B` absorbed the three contract commits (`6e4fb08d` handle, `ef6f2a16`    |
| views, `27b1f48c` its note) plus `ed402137` — they define the block-author and buffer ABIs the seam exposes, so they    |
| must land WITH the seam, not after it. `PR-G` grew by four: the algorithm-layer split, the unified block, its analytic  |
| tests and the rename follow-through — and it is now the largest unit at 16, worth splitting into `PR-G1` engines and    |
| `PR-G2` the unified block if reviewers baulk. `PR-I` gained the tag/settings tests and the `blocks/device` fold.        |
| **Still true, and the reason to build forward rather than reorder in place:** the units are NOT topologically ordered — |
| `PR-C`/`PR-D`/`PR-G` all sit on the `PR-B` seam and several test commits reference blocks introduced later.             |

### Performance record — HISTORICAL (measures `FFT2`, a block that no longer exists; the unification deleted it)

> Kept for the SHAPE it establishes — device throughput scales with edge buffer size while host degrades on the same
> buffers, so the knob is two-sided and stays per-edge — and for the nsys counts below, which are still the best
> evidence that the boundary is transport-bound. For current numbers use §35 (idle-card, all backends) and §30
> (framework cost). Original heading: "Authoritative performance record — warm, interleaved, spread reported".
> `host source -> FFT2 -> host sink`, only edge buffer + domain vary; 3 warm-up rounds discarded, 5 interleaved measured
> rounds, 4 Mi samples/run (`rebaseline.cpp`):

| edge buffer | host MS/s (min-max) | gpu:sycl MS/s (min-max) | gpu/host |
| ----------- | ------------------- | ----------------------- | -------- |
| 0.5 MiB     | 114.17 (89.8-116.5) | 10.40 (9.8-10.8)        | 0.09     |
| 2 MiB       | 107.02 (84.4-108.7) | 11.88 (11.3-12.9)       | 0.11     |
| 8 MiB       | 77.00 (65.7-82.9)   | 23.60 (22.1-24.6)       | 0.31     |
| 32 MiB      | 44.42 (38.2-46.2)   | 36.74 (31.5-43.5)       | **0.83** |

Device scales **3.5x** with buffer size, **still climbing at 32 MiB** (no knee). Host **degrades 2.6x** on the same buffers,
so the knob is two-sided and stays per-edge. The device column is a **FLOOR** — interleaving lets the GPU idle during host
runs; a sustained GPU-only warm run of the 8 MiB case reaches **57-67 MS/s**.
**nsys counts (robust; its timings were cold and retracted):** kernels **0.5% of wall**, GPU **idle 72%**, managed USM the
host touches is fault-migrated **6113 times, median 4 KB, 260 MB for 64 MB of data (4x)**. Also: `pollDeviceError()` does a
full `queue->wait()` at the top of **every** `dispatch()` — demoting that to a poison-flag check is small and immediate.

### MEASUREMENT PROTOCOL — MANDATORY. Ten claims died today for want of it.

1. **Pin the GPU clock**: `sudo nvidia-smi -lgc 1500,1500`, release with `-rgc`. Needs root and a real TTY — ask the user;
   `sudo` has no TTY here and the `!` prefix does not give it one. Unpinned the card parks at **210 MHz of 2100**.
2. **acpp's JIT/kernel cache warms across PROCESSES** (`ACPP_APPDB_DIR`): an unchanged binary climbed 57->67 MS/s over four
   runs, first cold run 26.9. **Discard warm-up runs.**
3. **Interleave A/B inside one process or round**; never compare absolutes across probes or machine states.
4. **Report median + min/max**; an effect smaller than the spread is _unresolved_, not small.
5. **Keep an internal control** (the host column moves only if the machine moved).
6. **Prefer counts to wall-clock subtraction** — nsys fault/launch counts survive warming; in-order-queue phase splits do not.

### Landed (all gated gcc15 + clang20 `-O2 -Werror`, Emscripten where applicable)

1. `fix(core)` **buffer leak** — `do_deallocate` unmapped half of what `do_allocate` mapped; 150 cycles of a 1 MiB ring leaked
   150 MiB of shmem, invisible in RSS. One line + regression test. **The unambiguous win.**
2. `feat(core)` capability registry `{usesMMAP, deviceOnly}`, no RTTI/exceptions/backend headers, unknown resource reads back
   as plain host memory. 272 asserts.
3. `feat(core)` `CircularBuffer` device-only support — skips host element-construct and wrap mirror; refuses device-only with
   non-trivially-copyable T, and device-only without double mapping.
4. **CUDA-VMM device-only resource**, 2 MiB granularity floor, explicit `-DGR_ENABLE_CUDA_VMM`.
5. `feat(device)` **device-only interior edges** — 1.45-1.47x warm; the wrap-mirror alternative was A/B'd and ruled out.
6. `feat(basic)` **transfer blocks really transfer** (bulk queue memcpy; were host `ranges::copy` placebos). No gain — capability.
7. Four measurement harnesses. (`qa_FftCrossover` was among them and has since been dropped — see the note in the
   READ FIRST block; the benchmarks `bm_FFT_backends` and `bm_fft_stream_graph` remain.)
8. `build(device)` GL explicit opt-in + dead-surface removal (this session's pre-work).

### CLOSED — do not re-open (each killed by measurement)

device-aware wrap copy (0.27 ns of 93) · pinned-**in-place** edges (~1.16x; correct physics — zero-copy means every FFT pass
streams over PCIe) · staging inside FFT2 with managed edges (no change) · the wrap mirror as the GPU bottleneck (A/B refuted)
· GPU clock as the _cause_ (it was the instability, not the cost) · per-call submission as the _floor_ (bounded at ~4
ns/sample by the staged probe) · the 18.4%-of-wall copy attribution (retracted) · "8 MiB saturation" (withdrawn) · **the
strict `isDeviceOnly` predicate** — its "-35% regression" was a cold artefact and a warm re-run on the case it exists for
(multi-pass const `processBulk` on a host-accessible edge, `predicate_probe.cpp`) shows **no difference**; keep
`isDeviceAccessible` · **the heap corruption** — does NOT reproduce; ASan on the real failing config exits 0 with no finding;
it was in reverted FFT2-scratch WIP, not committed code. Also: pinned edges collapse `host:sycl` to 0.7 MS/s (vs ~160), so
pinned memory is only ever a single-touch DMA endpoint · single-FFT-graph supremacy over the CPU is lost on physics.

### OPEN — HEAVILY STALE, re-verified 2026-08-21/22. Read the READ FIRST block's "Still open" list instead.

> Closed since this list was written, verified against the code: the **`DeviceSpans.hpp` @brief** (already written,
> `:18-20`), **F2/F9/F10** (see §18.2), the **CPU-fallback tag drop** (closed by maintainer decision, §36.1), the
> **PR mapping** (now 72 commits, not 55), and **`qa_FFTPerformance` never having run** (root-caused, §36.3 — one
> CMake flag). What remains genuinely open is listed in the READ FIRST block.

- **PR re-cut NOT STARTED, and its mapping is STALE** — the table below counts 47 commits and predates the FFT
  unification (`fft.hpp` rewritten, `FFT2` gone) plus `6e4fb08d`/`ef6f2a16`. **Re-derive against the current 55 before
  executing.** Method settled: build forward from `origin/main`, `cherry-pick -n` per unit + one commit each, build-gate
  every unit, `git branch -f` only at the end. Backup `backup/pre-pr-recut`.
- **F1 — device-side mutation is still not a compile error** (self-declared merge blocker). The canary runs in all
  builds but `DeviceProbeSafe` excludes any pmr-owning block, so a `mutable` pmr member still slips through. A true
  compile-time refusal needs C++26 reflection.
- **CPU-fallback drops a published tag** (self-declared merge blocker) — no decision, no test asserting either
  semantics. Bug or intended?
- **`DeviceSpans.hpp` @brief needs one line**: `requires { in.rawTags(); }` used to be TRUE (declaration existed, the
  static_assert fired only on use) and is now FALSE. A duck-typed `processBulk` guarding tags with `if constexpr` will
  silently skip that branch on a device and run it on the host — the divergence `ed402137` closed, reopened via
  feature-testing. No in-tree block does this; downstream-facing.
- **`qa_FFTPerformance` + the acpp smoke target have NEVER been run** — `build-acpp` cannot enable the block registry
  (`python3-devel` absent; the block-lib generator fails to regenerate). They are the only place the new
  `requires std::same_as<T, float>` on the stream SYCL hatch is exercised. A fresh acpp build dir would fix it.
- **complex→complex STREAM mode was deliberately dropped** in the unification (`qa_FFT.cpp:181` asserts it must not
  compile). Complex input is spectrum-only now — a capability removal to state at merge.
- **`sample_rate` rescale audit**: other DataSet-producing resampling blocks (`StreamToDataSet` …) have no
  `forwardSettings` opt-out and likely carry the same axis corruption the FFT block just fixed.
- **SimdFFT `Order::Unordered` unreachable** — `gr::algorithm::FFT` hardcodes `Ordered` on every path; measured 12-26%
  left unclaimed.
- `host:sycl` figures **embargoed**: ±15% run-to-run on an unchanged binary (149.4/192.2/156.9).
- **Re3 ANSWERED** (do not redo): FFT and FFT2 were never comparable — different output types and resampling contracts
  — and already shared one engine (bit-identical, 0.0 abs/rel error). Superseded by the unification.
- **Re4 ANSWERED** (do not redo): keep `GLSL2WGSL` + `ShaderFusion` (maintainer decision — WebGPU seed); GL compute can
  never reach a browser; substrate decided = emdawnwebgpu + `webgpu.h`. See the WebGPU section above.

### Env

Probes in the session scratchpad: `rebaseline` `crossover_probe` `strategy_probe` `bulk_probe` `migration_probe`
`mirror_probe` `side_probe` `quantum_probe` `staged_probe` `prof_target` `predicate_probe` `pinned_repro` `small_pinned`
`asan_target` `leak_probe` `touch_probe` `vmm_probe` (+ `wip-staged-boundary.patch`, `syclExperiments-pre-rewrite.bundle`).
Standalone acpp builds: take flags from `build-acpp/compile_commands.json`; link `libgnuradio-core.a`,
`libgnuradio-blocklib-core.so`, `libexprtk.a`, `libcpr`, `libcurl`, `-lcuda -lEGL -lOpenGL`. CUDA driver headers at
`/usr/local/cuda-13.2/targets/x86_64-linux/include`. `clang-format-18` is the CI formatter. Patched-header-copy trick for
experiments without touching the repo: symlink farm + `-I` override (see `asan_target`/`predicate_probe` recipes).

**Map of this doc:** this block is the current state. Below: §D1-§D3 device-edge design + step-0 hardware results (valid) ·
§1-§10 architecture reference · §11-§14 generic/GPU open work · §15 gotchas · §18.1/§18.2 open locked decisions and fix list.
Where a later note contradicts an earlier one, **the later note wins**.

## 0. Process constraints

All implementation follows the CLAUDE.md style guide plus these constraints:

1. **AI-assisted workflow.** Claude Code implements; the user guides, reviews, and approves each step.
2. **Review gate between stages.** Each stage/commit ends with a review + Q&A. Open points from the
   current and next stage are discussed. Advance only after an explicit **"go ahead"** from the user.
3. **No autonomous commits or pushes.** Claude Code does not `git commit`/`git push` (nor `--amend`,
   rebase, or force-push) unless the user explicitly authorises it, single-use. Work is presented as
   uncommitted changes for review. Commits carry `Signed-off-by: Ralph J. Steinhagen
<r.steinhagen@gsi.de>` and **no** co-author trailers. Local builds use `-j6` (max 6 cores).
4. **Each step is testable.** Every commit must compile and pass its tests before the review gate.
5. **Agent budget — HARD LIMIT (user directive 2026-08-17, after three workflows burned ~1.9M subagent
   tokens and hit the session limit twice).**
   - **At most ONE Opus/Fable-based agent running at a time.** No parallel fan-out of reasoning agents,
     no judge panels, no multi-proposal tournaments. Sequential only.
   - **At most 4-5 Sonnet-based agents**, and only for **mechanical code refactoring** — not for design,
     analysis, or evaluation.
   - Design/analysis work is done by the orchestrator directly. Prefer a targeted grep or a one-file read
     over delegating; most "probe" questions here are answerable in two shell commands.
   - This overrides any ultracode/workflow default that encourages breadth. If a task seems to need a
     fan-out, narrow the question instead.
6. **Nomen est omen — the name carries the meaning, not a comment above it (user directive 2026-08-25).**
   Generated code here has been consistently over-documented: comments that explain _reasoning_ sit in the
   source, which is the wrong place for them. The source is the single source of truth about _behaviour_, and
   a comment that restates or rationalises behaviour is a second copy that drifts out of step with the first.
   The doc-vs-code contradiction is not hypothetical — this branch shipped a runtime warning naming a hatch
   the framework no longer had, and an invented example value on a getter that returns something else.
   - **Name it instead of explaining it.** Methods, free functions, lambdas, parameters, variables and
     types take names that state their functional use. If a comment is needed to say _what_ something does,
     the name is wrong — rename it and delete the comment.
   - **Reasoning belongs outside the source**: the commit message, this file, or a design doc. Keep in-source
     comments to what a name genuinely cannot carry — a unit, a valid range, a regulatory or hardware
     constraint, a workaround and why it exists. One line, not a paragraph.
   - **Do not narrate the obvious**, do not restate the signature in prose, and do not leave a comment
     describing code that has been removed.
7. **Editing by line range: anchor on syntax, then verify balance (earned 2026-08-25, three times).**
   Bulk edits driven by `(file, startLine, endLine, replacement)` specs broke the tree three times in one
   session: a range swallowed the `};` closing a struct, another replaced a line _inside_ a function with the
   function's own signature, and a grep anchor matched a phrase that a previous edit had also written into a
   `@brief` fifty lines away. Two of the three surfaced as errors deep inside CUDA and libstdc++ headers
   (`using declaration in class refers into 'std'`), which points at the toolchain and not at the edit.
   - **Anchor a range on the declaration line, then walk brace depth to find its end** — never on comment
     prose, which repeats.
   - **Run a `{}` / `()` / `/* */` balance check on every edited file before building.** It catches the whole
     class in a second; a build catches it in four minutes and blames the wrong file.
   - **Strip ANSI codes before grepping a build log for `error:`.** A `grep -c ' error:'` silently reported
     zero while the log held two genuine errors, because the colour escape sits between the space and `error`.

8. **Extract to a lambda first, to a free function only when it earns it (user directive 2026-08-25).**
   A long method is not by itself a reason to split it into several named methods — that inflates the
   user-facing API for an internal problem.
   - Prefer an **in-method lambda**: expressive, named, scoped to the one place it is used, invisible in
     the API.
   - Promote to a **free-standing function only when all three hold**: it is generic, it is unit-testable
     on its own, and it has callers outside the method that spawned it. Two of three is not enough.

---

## 1. Executive summary

Integrate heterogeneous compute (GPU/FPGA/accelerator) into GR4. Primary backend: SYCL via
AdaptiveCpp. WASM fallback: GL compute (GLSL→WGSL). Native CUDA/ROCm are future extension points.

**Core principle.** A user who writes `processOne(...) const noexcept` or `processBulk(...)` for
CPU runs the same code on a device unchanged at ~70–80 % of device peak. Escape hatches
(`processBulk_sycl`, `shaderFragment`) let experts hit 100 % with native, type-safe parameters.

Two use-cases: **(A)** standalone device blocks (transfer in, compute, transfer out) and
**(B)** device sub-graphs where data stays on-device between blocks (Phase 2).

## 2. Goals and constraints

1. Preserve the Block API — `processOne`/`processBulk` unchanged.
2. `const noexcept processOne` auto-parallelises onto any backend (same gate as SIMD).
3. Backend-specific escape hatches with native parameter types — zero-cost per backend.
4. Backend-agnostic selection via the existing `compute_domain` string + `ComputeRegistry`.
5. **Zero cost for CPU-only graphs** — device paths are compile-guarded; CPU+SIMD dispatch untouched.
6. CI-testable without GPU hardware — AdaptiveCpp CPU/OpenMP fallback; Mesa llvmpipe for GL.
7. **Composition over modification** — device dispatch lives in a composed `ExecutionStrategy`;
   `Block<T>` gains one seam, not hundreds of lines.

Constraints: **primary GPU target NVIDIA** (AMD/Intel via the SYCL abstraction); **no external GPU
library dependencies beyond AdaptiveCpp** (SSCP, clang ≥ 20) for Phase 1; float32 focus initially with `gr::complex<float>` for complex blocks;
SIMD retained on CPU (`vir::simd`); GCC 15 / Clang 20 / Emscripten must build cleanly with GPU **off**.

## 3. Dispatch hierarchy & escape hatches

Selected at compile time from block traits + `compute_domain` + active backend:

```
1. SYCL backend  + processBulk_sycl   → call processBulk_sycl(sycl::queue&, …)      (native escape hatch)
2. GLSL/WGSL     + shaderFragment      → compile shader → bind SSBOs → dispatch       (shader escape hatch)
3. CUDA/ROCm     + processBulk_cuda/…  → native stream                                (future)
4. any device    + const noexcept processOne → auto-parallelise per backend           (zero-effort path)
5. CPU domain    + const + SIMD        → invokeProcessOneSimd   (existing, untouched)
6. CPU domain    + const, no SIMD      → invokeProcessOnePure   (existing, untouched)
7. CPU domain    + non-const           → invokeProcessOneNonConst(existing, untouched)
8. processBulk                         → existing, untouched
```

Levels 5–8 are unchanged; the device branch is inserted _before_ level 5 via a composed strategy.
Two escape-hatch families: **pointer-based** (SYCL/CUDA/ROCm — USM pointers) and **shader-based**
(GLSL/WGSL — buffer handles, not pointers; gotcha G9). A block may provide several at once.

**Trait gates** (`BlockTraits.hpp` / `device/ExecutionStrategy.hpp`, actual form):

```cpp
concept AutoParallelisable = HasConstProcessOneFunction<D> && HasNoexceptProcessOneFunction<D>; // == SIMD gate
concept HasSyclBulk        = false;  // address-of-member false-negatives templated overloads; kept as a stub
concept HasShaderFragment  = requires(const D& b) { b.shaderFragment(); };
concept DeviceEligible     = AutoParallelisable<D> || HasSyclBulk<D> || HasShaderFragment<D>;
// the real probe is an expression against the actual queue/span types, evaluated only in SYCL builds:
concept HasSyclBulkForSpans = requires(SyclQueue& q, TBlock& b, InputSpans& in, OutputSpans& out) { … };
```

**Naming convention** for specialised process functions:

| Function                             | Backend             | Parameters                                                  |
| ------------------------------------ | ------------------- | ----------------------------------------------------------- |
| `processOne(T) const noexcept`       | all (auto-parallel) | scalar value                                                |
| `processOne(simd<T>) const noexcept` | CPU SIMD            | SIMD vector                                                 |
| `processBulk(…)`                     | CPU                 | `InputSpanLike`, `OutputSpanLike`                           |
| `processBulk_sycl(sycl::queue&, …)`  | SYCL                | native queue + spans over USM                               |
| `processBulk_cuda(CUstream, …)`      | CUDA (future)       | native stream + device spans                                |
| `shaderFragment()`                   | GLSL/WGSL           | returns `ShaderFragment` (source + constants + chunk sizes) |

The `_sycl` suffix is kept (not an overload on the first parameter) so the escape hatch is greppable and the
trait probe names one member. `HasSyclBulk` itself is `false`: a templated `processBulk_sycl` (FFT2 takes `auto&`
spans) makes `&D::processBulk_sycl` ill-formed, so `ExecutionStrategy` probes callability against the real span
types instead — `Block.hpp` asks `ExecutionStrategy<D>::canDispatch<InputSpans, OutputSpans>()`.

## 4. Composition architecture

`gr::device::ExecutionStrategy<TBlock>::dispatch(block, in, out, count, computeDomain)` resolves a `DeviceContext`
from the domain, then `if constexpr` selects the path: `processBulk_sycl` (H2D → call → D2H) → `shaderFragment`
(compile/cache → SSBO → dispatch) → auto-parallel (`parallelFor` over `processOne`). It owns H2D/D2H transfer and
tag forwarding.

```
user block:  processOne(T) const noexcept │ processBulk_sycl(queue&,…) │ shaderFragment()
                                  ↓
Block<T>::dispatchProcessing()  — if constexpr(DeviceEligible && canDispatch) → ExecutionStrategy
                                  ↓                       else → existing SIMD/scalar dispatch (untouched)
ExecutionStrategy<TBlock>        — H2D/D2H • tag forward • path select
                                  ↓
DeviceContext (virtual)          — DeviceContextCpu │ DeviceContextSycl │ DeviceContextGLSL │ (CUDA/ROCm future)
```

The seam is **generic**: "select an execution strategy from `compute_domain`", not a GPU-only branch — CPU/MCU
specialisations plug into the same point. Device blocks with no registered backend log once and fall back to CPU.

## 6. Memory model

The PMR chain is already in `origin/main` (A): `ComputeDomain` → `ComputeRegistry::resolve()` →
`memory_resource*` → `EdgeParameters` → `Port::resizeBuffer()` → `CircularBuffer(size, allocator)`.

B adds `gpu::UsmMemoryResource : std::pmr::memory_resource` wrapping `sycl::malloc_shared`/`sycl::free`
(CPU fallback: `operator new`/`delete`), registered via `registerUsmProvider()` for `"gpu:…"` domains.
Once registered, `CircularBuffer`/`Tag`/`Tensor`/`Value` get device-accessible memory through the
existing plumbing. `registerSyclRuntime()` calls `registerUsmProvider()` as part of runtime registration
(`device/SyclRuntime.hpp:102`), so device edges allocate through USM without per-test setup.

- **Atomic counters and USM are incompatible** across CPU/GPU — the same atomic cannot be coherently
  accessed from both. Explicit transfers are always required at CPU↔device boundaries;
  `CircularBuffer` cannot transparently bridge host↔device.

**`DeviceContext`** — virtual base, user-extensible; backend known at compile time by the strategy:

```cpp
struct DeviceContext {
    virtual DeviceBackend backend() const noexcept = 0;
    virtual DeviceType    deviceType() const noexcept = 0;        // CPU/GPU/FPGA/Accelerator
    virtual std::string   shortName() const = 0;  // "CPU", "SYCL:RTX 3070", "GLSL:llvmpipe"
    virtual std::string   name()  const = 0;       // "NVIDIA GeForce RTX 3070"
    virtual std::string   version() const = 0;
    template<class T> T* allocateDevice/allocateShared/allocateHost(std::size_t n);  // typed wrappers
    virtual void copyHostToDevice(const void*, void*, std::size_t) = 0;              // + copyDeviceToHost, wait
    virtual void* allocate{Device,Host,Shared}Raw(std::size_t bytes, std::size_t align) = 0; // + deallocateRaw
};
// DeviceContextCpu (heap+memcpy) │ DeviceContextSycl (USM+parallelFor) │ DeviceContextGLSL (SSBOs+dispatch)
```

Derived classes must `using DeviceContext::copyHostToDevice;` to un-hide the typed template wrappers,
else typed calls silently bind the `void*` overload and treat element-count as bytes (gotcha G17).

**`DeviceBuffer`** — RAII/move-only; `SharedDeviceBuffer = shared_ptr<DeviceBuffer>` for fused blocks;
`DeviceBufferRegistry` tracks lifetime (use-after-free in debug), false-sharing, total device memory.

## 7. `gr::complex<T>`

`std::complex<T>` is **not a portable/reliable SYCL device-kernel type** under AdaptiveCpp/hipSYCL-style
backends — not because it can't be transferred (it is trivially-copyable, adjacent re/im on modern
libstdc++/libc++, fine for host storage and USM byte transport) but because its **arithmetic** is
unreliable in device code (see G8 + AdaptiveCpp #340/#341). `gr::complex<T>` (`Complex.hpp`) is a
`constexpr`, trivially-copyable `{re, im}` struct with device-safe inline arithmetic + SIMD/tuple/ADL
integration, layout-compatible with `std::complex<T>` at controlled ABI boundaries:

```cpp
template<class T> struct complex {
    T re{}, im{};
    constexpr complex(const std::complex<T>& c) : re(c.real()), im(c.imag()) {} // implicit, both ways
    constexpr operator std::complex<T>() const { return {re, im}; }
    friend constexpr complex operator*(complex a, complex b) { return {a.re*b.re-a.im*b.im, a.re*b.im+a.im*b.re}; }
    // +,-,/, real/imag; structured-binding + vir::simdize support; gr::real/imag/abs/norm/conj free fns (ADL)
};
static_assert(sizeof(complex<T>) == sizeof(std::complex<T>)); // reinterpret_cast-safe
```

> **Bottom line** (full rationale in the type's `@brief` + gotcha G8): `gr::complex<T>` is justified for portable
> device/SIMD _arithmetic_, not because modern `std::complex<T>` is unsuitable for host storage or USM byte
> transport. ABI caveat: the `sizeof`/`alignof` guards do **not** bless arbitrary
> `reinterpret_cast<std::complex<T>*> ↔ gr::complex<T>*`; treat it as a deliberately isolated bridge.

**Integration status:** the structural `meta::complex_like` keystone landed and carried `TensorMath::real/imag/conj`,
`squaredMagnitude` (which silently returned `z²` for a `gr::complex`), `SVD` and `DataSetUtils` with it. Still
`std::complex`-only: the **wire/identity tier** — `pmt::is_complex` (`PmtTypeHelpers.hpp:41/44`),
`Value::ValueScalarType`, `ValueHelper`, `ValueMap`, `YamlPmt`, `formatter`, `portableTypeMapping` — plus
`UncertainValue` (needs `gr::pow/log/hypot`). See §12.3.

## 8. Shader / GLSL / WGSL backend

> **HISTORICAL — the shader backends left core under D5 (§50, §55, §57). Kept for the reasoning, not as a
> description of the tree.**

Blocks describe a kernel as GLSL source via:

```cpp
struct ShaderFragment {
    std::string              glslFunction;    // "float process(float x){ return x*GAIN; }"
    std::vector<ShaderConst> constants;       // baked: {{"GAIN", gain}}
    std::size_t              inputChunkSize;  // 0 = element-wise, N = needs N-sample chunks (e.g. FFT)
    std::size_t              outputChunkSize;
};
```

The runtime compiles, caches (`ShaderCache`, LRU, hash-keyed by source+backend), binds input/output
SSBOs from port data, and dispatches.

**`ShaderFusion`** fuses adjacent shader blocks in the same domain:

- _element-wise chain_ (all `inputChunkSize==0`): inline-compose `out[i]=fC(fB(fA(in[i])))` — one
  dispatch, no barriers.
- _bulk block in chain_ (FFT, `inputChunkSize==N`): multi-stage dispatch with barriers — element-wise
  pre over k·N, barrier, bulk over k chunks, barrier, element-wise post; input chunked to LCM size.
- settings change → invalidate the fused shader (recompile via `ShaderCache`).

**Runtime specialisation** (the key differentiator over SYCL's build-time SSCP): per-block, settings
split into **baked constants** (embedded in source; recompile on change — `fft_size`, twiddles) vs
**uniforms** (uniform buffer; update without recompile — `scale`, `threshold`). Example:

**Dispatch paths, one shader model:** native GL 4.3+ compute + SSBOs (current, EGL headless, CI on
Mesa llvmpipe); GL 3.0 / WebGL2 transform-feedback fallback; WebGPU/WGSL via `GLSL2WGSL` (future).
SSBOs are native-only. `GR_HAS_GL_COMPUTE` (currently auto-detected — see §16) guards
`GlComputeContext`/`DeviceContextGLSL`. WASM init uses `emscripten_webgl_create_context` instead of EGL.

## 9. FFT implementations + measured performance

Both GPU paths use **Van Loan Stockham** radix-2 auto-sort (Van Loan, _Computational Frameworks for
the FFT_, ch. 2): no bit-reversal pass, output in natural DFT order; sequential reads (`srcLo=j`),
interleaved writes (`dstLo=group·Ls+k`); butterfly `dst[lo]=a+w·b`, `dst[hi]=a−w·b` (twiddle on `b`
before the sum); out-of-place ping-pong. `SyclFFT` (SYCL; CPU via `SimdFFT` delegation) and `GlslFFT`
(GLSL compute) share the index mapping, validated by `forwardStockhamCpu()` against `SimdFFT` — **no
GPU hardware needed** for algorithm tests.

**Execution phases (Stockham):**

- _global stages_ (half-span > workgroup): one kernel per stage, all batches fused into one launch
  (`sycl::range<2>{nBatches, N/2}`); read `src` → write `dst` → swap.
- _fused local-memory stages_: a single kernel loads a tile into `local_accessor`, runs the remaining
  butterfly stages with `group_barrier`, writes back; the workgroup's twiddle sub-table is loaded into
  shared memory once. Transition: `sLocal = max(0, log₂N − log₂(2·wgSize))`. All stages chained by
  `sycl::event`, single host wait.

| wgSize | shared (data+tw) | local stages | global stages (N=65536) | launches |
| ------ | ---------------- | ------------ | ----------------------- | -------- |
| 256    | 6 KB             | 9            | 7                       | 8        |
| 512    | 12 KB            | 10           | 6                       | 7        |
| 1024   | 24 KB            | 11           | 5                       | 6        |

**Kernel arithmetic rules (mandatory):** `uint32_t` + bit-shifts — 64-bit div/mod is ~100+ cycles
(emulated) on GA104 and **was 96 % of kernel time**; `/halfSpan → >>log2`, `%halfSpan → &(halfSpan−1)`;
batch index from `range<2>` (free). Optional radix-4/8 per-thread fuses `log₂R` stages into registers.

Measured (RTX 3070, GA104; memory roofline ~93 GFLOP/s, achievable BW 405 GB/s):

| Version | Description                    | x128 N=65536   | % roofline | vs CPU SimdFFT |
| ------- | ------------------------------ | -------------- | ---------- | -------------- |
| V0      | DIF, sync `.wait()`            | 2.3 GFLOP/s    | 2 %        | 0.16×          |
| V1      | DIF, async events              | 11.3 GFLOP/s   | 12 %       | 0.82×          |
| V2      | Stockham, uint32               | 12 GFLOP/s     | 13 %       | 0.82×          |
| V2+10   | V2, benchmark<10> (JIT warmup) | 51 GFLOP/s     | 55 %       | **3.8×**       |
| V3      | radix-4 + warmup + SimdFFT CPU | **59 GFLOP/s** | **63 %**   | **3.2×**       |

acpp SSCP cold-start ~14 ms/kernel, amortises to ~70 µs sustained → benchmark with JIT warmup, not
single-shot. L2 cliff: x16 N=8192 (fits L2) 36 GFLOP/s vs x128 N=65536 (GDDR6) 59 GFLOP/s.
Tests: `qa_FFT2.cpp` — 7 suites / 37 tests. Gaps: R2C/C2R, Bluestein (non-pow2), async triple-buffer
streaming, sub-group-shuffle (+22 %, validated, not integrated).

## 10. Blocks

**Transfer blocks** (`blocks/basic/.../TransferBlocks.hpp`) mark CPU↔device boundaries and cap DMA size; between an
`HostToDevice`/`DeviceToHost` pair data stays on-device (USM edges). `chunk_size` is a per-call cap bounded by
`[min_chunk_size, max_chunk_size]`, adaptively doubled/halved by a throughput estimator. Tags at a sample boundary
split the chunk (apply settings, then continue); informational tags pass through.

**`FFT2<T>`** (`gr::blocks::fourier`) — raw forward/inverse FFT over `gr::complex<T>` ports, one type dispatched to
CPU or GPU by `compute_domain`: `processBulk` delegates to `SimdFFT` (zero-copy `reinterpret_cast`),
`processBulk_sycl` to `SyclFFT`, `generateShader()` to the GLSL path; `settingsChanged` re-inits both FFTs,
invalidates the shader, and updates `input_chunk_size`. Raw `gr::complex` output (not `DataSet`) keeps it
composable — windowing/magnitude/phase are separate blocks. Coexists with the old `gr::blocks::fft::FFT`.

## 11. Diagnostics / unified error+log+profiler channel

Design of record: `featDiagnosticChannel_design.md` (§14, in particular §14.8). Three faces of one mechanism:
synchronous `std::expected<T, gr::Error>` returns + canonical `gr::log::LogRecord` for out-of-band/device-emitted
diagnostics + a `boundary()` exception→value catch-all (still open, §12.4). Do not resurrect the removed
`Diagnostics.hpp`/`DiagnosticsRing.hpp` parallel implementation.

**Shipped** (`c14ba40a`): `gr::log` front-end with a self-flushing (write-through) `FixedRecordBackend`; **no runtime
level filter** — `fatal/failure/error/warning` always emit, `info/debug/trace` are gated only by compile-time
`kDebugBuild`; the C ABI is gone. Device logging rides `ValueMap`: `DeviceLog.hpp` gives a POD `DeviceLogSlab`
(atomic claim-or-drop), `StaticDeviceLogSlab<N>` for MCU, and a `DeviceLogger` with the host's call syntax
(`log.warning("processed {} items", i)`), rendered host-side with real `std::format`;
`device/DeviceLoggerBackend.hpp` owns the slab through `DeviceContext::allocateShared` and merges it into the host
backend on `flush()` **after** `wait()` (cross-PCIe coherence). Two additive `ValueMap` changes enable it:
`ValueMapView::formatAt` and a libc-free `detail::keyEquals` (see G19). Tests: `qa_DeviceLog` (always built) and
`qa_DeviceLoggerBackend` (acpp-only, gated in CMake). Docs: `docs/USER_API_Logging.md`.

---

## 12. Generic infrastructure still required (MCU / CPU / GPU)

**12.1 Generic execution-strategy seam in `Block` — DONE** (`Block.hpp:1847`, `DeviceEligible`-gated; CPU-only graphs
are bit-for-bit unchanged and pay nothing).

**12.2 Freestanding/MCU-clean execution layer — OPEN.** `execution.hpp` has 5 `try/catch` sites and no
`__cpp_exceptions` guard: a hard error under `-fno-exceptions`. Guard the receiver adapters, hosted-guard the
blocking terminals (`sync_wait`, `when_all`), keep `bulk` freestanding.

**12.3 `gr::complex` wire/identity tier — OPEN** (own PR; see §7). Decide consciously: first-class everywhere, or
fixed as a device-/wire-boundary type. The "partial" state is a latent trap.

**12.4 Generic error propagation through the seam — OPEN.** Alloc failure, shader-compile error, device-lost,
trailing samples → `std::expected`/tags/`requestStop()`, never a silent fallback; exception-free (MCU). Remaining:
`boundary(...)`, the `std::expected<work::Status, gr::Error>` collapse, and an optional in-kernel USM status
accumulator drained after `wait()`. Any POD status vocabulary must feed `gr::log`/`gr::Error` — it must not
reintroduce a parallel diagnostics ring. Design: `featDiagnosticChannel_design.md`.

**12.5 Generic rechunking — CLOSED**, nothing to extract (see "Decisions worth keeping").

**12.6 Generic settings-snapshot reflection — CLOSED** with `DeviceBlockState`'s deletion.

**12.7 `compute_domain` grammar — OPEN (docs).** Document `default_cpu`/`default_io`/`kind[:backend[:index]]` and
the fall-back-to-CPU-with-warning rule as a generic execution-selection contract.

## 13. Open engineering items (prioritised)

> **STALE (July 2026) — a record, not a work list.** Read the OPEN table in the READ FIRST block instead; every
> item here predates D5, the `Domain`→`SubGraph` rename and the history rebuild.

- [ ] **P1 — §12.4 error propagation**: `boundary(...)` + the `std::expected<work::Status, gr::Error>` seam, plus its
      tests (allocation failure, in-kernel error).
- [ ] **P1 — host drain loop for device logs.** Today the user calls `gr::log::flush()` after `context.wait()`. A
      larger application wants a dedicated thread notified on new records (console/tty/network backends).
- [ ] **P2 — `gr::complex` wire/identity tier** (§12.3).
- [ ] **P2 — GL: explicit opt-in.** `GR_HAS_GL_COMPUTE` still auto-enables wherever EGL+OpenGL dev packages exist
      (`CMakeLists.txt:207-219`), so the compiled surface changes per machine; `qa_Complex`, `qa_Execution` and
      `qa_UsmMemoryResource` are still added unconditionally (`core/test/CMakeLists.txt:67-69`). Gate both.
- [ ] **P2 — GLSL typed-buffer policy.** The non-float guard is safe; either constrain shader fragments to `float`
      by concept, or teach GLSL buffers/copies/shader generation about further stream value types.
- [ ] **P2 — scheduler work accounting.** Whether device dispatch should report explicit span counts: not needed to
      prevent publishing uncomputed samples, but useful for observability/partial-work reporting.
- [ ] **P3 — docs.** §12.7 grammar + `USER_API_GPU_Blocks.md` (`compute_domain`, `registerSyclRuntime()`, fallback,
      the same-source `qa_FFT2Device` pattern). (`docs/USER_API_Tag_mechanics.md` tag-ownership prose fixed in
      `31dc0e35` — standalone, cherry-pickable to main.)
- [ ] **P3 — profiler phase** (`featDiagnosticChannel_design.md` Phase 3: MCU/GPU-portable timing).

## 14. Remaining GPU-specific work

> **STALE (July 2026) — a record, not a work list.** Read the OPEN table in the READ FIRST block instead; every
> item here predates D5, the `Domain`→`SubGraph` rename and the history rebuild.

**Tier 1 (minimum viable wiring) — DONE**: seam, USM provider registration, domain→`DeviceContext` bridge, tag
forwarding, end-to-end tests (`qa_DeviceSeam`, `qa_FFT2Device`), and the GCC 15 / Clang 20 / Emscripten(GPU off) /
acpp matrix.

**Tier 2 — production integration.** STILL OPEN, re-verified 2026-08-22: automatic transfer-block insertion at
domain transitions (`makeDeviceSubGraph` now inserts them for a group; flat-graph auto-formation dropped, §38) · `processEpilogue` (dropped)
on the device path (0 references in `ExecutionStrategy`) · adaptive chunk auto-tuning · settings→shader
recompilation invalidation · `USER_API_GPU_Blocks.md`.
DONE since this was written: error reporting (§12.4/§18.1 decision 1, `dispatch` returns `std::expected`) and
`blocks/device/` folded away (`ef551b8f`). CI still pins `ACPP_TARGETS=omp`, so no GPU backend runs in CI.

**Tier 3 — first-class citizen.** GPU sub-graph scheduling (data stays on-device); scheduler-driven shader fusion;
`compute_domain` propagation (partial: `Graph.hpp:676` auto-populates an edge's domain from the block's
`compute_domain`; note the inverted edge-PMR precedence bug — `EdgeParameters` currently beats the Graph-ctor
`ResourceProfile`); ~~device-only buffers~~ (SHIPPED, PR-F) ; multi-device (`gpu:sycl:0` vs `:1`, F2 now done); profiling via tags;
~~WebGPU runtime backend~~ (SHIPPED, `3a1f96a9`); higher-radix/R2C/Bluestein FFT; full `gr::complex` migration (§12.3); `DeviceContextCUDA`/`ROCm`;
auto-GLSL from `processOne` (research).

## 15. Known gotchas (retain — hard-won)

| #   | Gotcha                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | Mitigation                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| --- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| G1  | `gr::complex` ↔ `std::complex` layout                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | `static_assert` on `sizeof`/offsets before any `reinterpret_cast`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| G8  | `std::complex` arithmetic unreliable in SYCL device kernels (a reliability, not transport, issue)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | `std::complex` multiply → libgcc `__mulsc3`, `std::abs` → libm `cabsf`: both unresolved at the CUDA JIT (acpp/clang 21 generic-SSCP, **verified**; `-ffast-math` inlines `__mulsc3`, not `cabsf`). AdaptiveCpp #340: silent-garbage multiply on ROCm (add/USM fine). Use `gr::complex` (inline ops + `std::sqrt` magnitude — device builtins); `std::complex` stays fine for host storage + USM transfer. Refs: AdaptiveCpp #340/#341 + §7.                                                                                                                                                                                                                                                                        |
| G9  | pointer escape hatches don't work for GL/WebGPU                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | GL/WGSL use buffer handles → must use `shaderFragment`, not `processBulk_sycl`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| G10 | **CORRECTED 2026-07-09 (measured).** acpp HCF kernel lookup fails when a kernel is launched _from a Boost.UT suite body_, because UT runs suites from `~runner` (static destruction) after acpp's HCF registry is gone: `omp_queue: Could not obtain hcf kernel info for kernel _Z18__acpp_sscp_kernelI...basic_parallel_for<K,1>`. Hits plain `parallel_for(range)`, not just `nd_range`. Isolated A/B: same kernel from a `main`-called fn = PASS, from a `"..."_test` in a global suite = ABORT. Second, separate hazard: in a TU containing kernels, a global `const suite<...>` object's ctor is silently dropped → `0 asserts in 0 tests`, **exit 0** (silent pass).                                                                              | **Fix:** register/run the tests from `main()` (`detail::test{"test", name} = [...]` for dynamic names). Kernels then launch while the runtime is alive; UT still owns the exit code (verified: failing assert ⇒ 255). A SHARED lib is NOT required. `cfg<override>.run()` from main also works, but does not cure the dropped-suite hazard.                                                                                                                                                                                                                                                                                                                                                                        |
| G12 | per-stage kernel launch overhead                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        | chain stages via `sycl::event` + in-order queue, single host wait (~70 µs/launch residual).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| G14 | 64-bit int div/mod in kernels                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | `uint32_t` + bit-shifts for pow2 dims; **was 96 % of kernel time**.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| G15 | DIF bit-reversal on GPU                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | Stockham auto-sort (natural order, no permutation); keep DIF for CPU.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| G16 | Stockham index mapping                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | Van Loan canonical (`srcLo=j`, `dstLo=group·Ls+k`, `a±w·b`); validated by `forwardStockhamCpu()`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| G17 | `DeviceContext` typed-template name-hiding                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              | `using DeviceContext::copyHostToDevice;` in every derived class.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| G18 | `GlComputeContext` dtor segfault                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        | `eglTerminate` invalidates the context; leaked Meyer's singleton for the GL context in test code.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| G20 | `NamedPortCollection::name` is a `std::string_view` (`BlockModel.hpp:203`), so a port name materialised into a **local** `std::string` dangles the moment the registration lambda returns. Symptom is remote from the cause: `dynamicPortFromName` compares freed memory, every dynamic port fails to resolve (`Invalid name specified name=output#0, base=output`), edges never connect, the scheduler never finishes, and the test **hangs to the ctest timeout**.                                                                                                                                                                                                                                                                                    | `Name` has static storage — take a `constexpr std::string_view` onto the `static constexpr fixed_string`, never re-materialise it into a local.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| G19 | **Any libc call reachable from device code is a landmine.** `std::string_view::operator==`/`starts_with(const char*)` lower to `memcmp`/`bcmp`/`strlen`. On the acpp **CPU/OpenMP** backend these only warn — `[LoopSplitterInlining] memcmp is not defined!` — because the pass cannot inline them but the linker resolves them against libc. On **CUDA** they either get folded by the optimiser (silently fine) or reach ptxas as `Unresolved extern function 'bcmp'` (hard fail). So the warnings are the _early_ symptom of the ptxas error: never ignore them. Measured: `keys::lookupId()` (`starts_with("gr:")` + `ranges::find` over `kCanonical`) produced 107×memcmp + 53×strlen; making it byte-wise took it to **0**, behaviour unchanged. | Keep everything on the device-callable `ValueMapView` find/try_emplace path libc-free (`gr::pmt::detail::keyEquals`). Treat a non-zero `LoopSplitterInlining` count as a build regression. A kernel with no ValueMap emits 0.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| G21 | **`Block::work()` is INERT for any block whose `blockCategory` is not `NormalBlock`.** `Block.hpp:2305-2311` short-circuits to `{requestedWork, 0UZ, OK}` before `workInternal`. A `ScheduledBlockGroup` IS in the parent's job list (`graph::flatten` defaults to `traverseCategory = TransparentBlockGroup`, so it adds the group but does not descend, `Graph.hpp:959-964`) and the parent's loop DOES call `work()` on it (`Scheduler.hpp:624`) — the call simply does nothing. Symptom if you assume otherwise: a hand-written group reports plausible `work::Result`s, no error anywhere, and zero samples move. This is also why today's `SchedulerWrapper<Simple>` is correct while its `_schedulerThread` does the real work.                  | A group that wants to be driven synchronously must **declare its own `work(std::size_t)`**, which shadows `Block<Derived>::work()` because `BlockWrapper::work()` forwards via a NON-virtual `blockRef().work(...)` (`BlockModel.hpp:772`). Do NOT relax the category check — it would wake every existing group. **Second half of the same trap:** `BlockWrapper::dynamicPortsLoader()` registers static ports ONLY for a `NormalBlock` (`BlockModel.hpp:690`), so a group's `PortIn`/`PortOut` members are invisible to `graph.connect` — boundary ports must come from `GraphWrapper::exportPort`. Both halves are reproduced on demand by `qa_DispatchGroup` (rename its `work()` ⇒ 0 samples move, silently). |
| G22 | **A device-only PMR resource can TERMINATE the process on a badly-sized allocation.** On a box with `GR_ENABLE_CUDA_VMM=ON`, a domain-interior edge resolves to `CudaVmmMemoryResource`, **not** plain SYCL USM — and its `do_allocate` calls `gr::log::fatal` (process-terminating) for any request that is not an exact multiple of its 2 MiB granule. A probe allocating `sizeof(float)` from an edge's resolved resource takes the whole test binary down, with no assertion failure to explain it.                                                                                                                                                                                                                                                 | Size any allocation from a resolved edge resource in whole granules: `gr::allocationGranularity(resource)` (`MemoryResourceCapabilities.hpp`), treating 0 as 'no constraint'. Also note `registerSyclRuntime()` mutates the process-global `ComputeRegistry`, so a test that calls it must run LAST or it changes what every earlier test observes.                                                                                                                                                                                                                                                                                                                                                                |

## 16. Open questions

> **STALE (July 2026) — a record, not a work list.** Read the OPEN table in the READ FIRST block instead; every
> item here predates D5, the `Domain`→`SubGraph` rename and the history rebuild.

1. ~~SYCL event overhead for many small blocks~~ — partly answered: a Domain drives its members synchronously on one
   thread, so there is no per-member event plumbing to overhead in the first place. An in-order queue per domain is
   still the direction for the async future.
2. Shader recompilation vs in-flight data on `fft_size` change — may need a pipeline drain.
3. CPU-SYCL FFT reaches ~3 GFLOP/s scalar vs SimdFFT ~20 — needs SIMD butterflies for parity.

## 17. Glossary & references

**SSCP** AdaptiveCpp single-source single-compiler-pass. **USM** unified shared memory.
**H2D/D2H** host↔device copy. **CPO** customisation-point object. **WGSL** WebGPU shading language.
**Baked constant** shader value compiled in (recompile on change); **uniform** updated without recompile.

External (Phase-2 study, full notes in git history): **TinyCompute** (Koen Samyn, CppCon 2025) —
kernel-as-struct (`local_size` + typed `BufferBinding<T,B>` + `main()` together; CPU runs `main()`
directly, GPU transpiles), `DimTraits` 1D/2D/3D dispatch, CPU/GPU parity via backend swap; we adopt
the patterns but not the Clang-AST transpiler (we use `ShaderFragment` strings).
**ComputeShadersTutorial** — ping-pong SSBO, `glMemoryBarrier` between passes, workgroup-size query.
**gpuAgnosticFunctor** (Karpowicz) — functor-as-kernel dispatched via `LaunchKernel(N, functor)` as the
minimal CUDA/SYCL/OpenMP primitive; reference for a future `DeviceContextCUDA`.

---

## 18. Next iteration — device integration design (2026-07-09)

Code-grounded constraints with file:line evidence live in **`featDeviceIntegration_findings.md`** (untracked).
Read that first; this section is the plan built on it.

### 18.0 Scope decision (user, 2026-07-09)

**Both** use-case A (standalone device blocks) hardened **and** use-case B (data stays on-device between adjacent
blocks) implemented. GLSL becomes a **first-class backend**, not a CI stand-in, even though its kernel-generating
callback differs from a host/SYCL C++ function.

### 18.1 Locked decisions

1. **Error channel.** `ExecutionStrategy::dispatch` returns `std::expected<work::Status, gr::Error>`, and the
   `DeviceContext` transfer/dispatch primitives stop being `void` (today a failed `copyHostToDevice` is
   unobservable). On failure the dispatch **early-returns** and writes `gr::log::{fatal,error,warning}` at the
   failure site. No separate in-kernel status word for v1: a kernel reports through the `DeviceLogger` slab.
   `gr::Error` allocates and is `static_assert`ed non-trivially-copyable, so it can never be the in-kernel type.
2. **Fixed in:out ratio for v1.** The device path assumes a fixed sample ratio and needs **no epilogue**; trailing
   samples are out of scope. This closes the partial-work reconciliation question — the device path may keep
   forcing `processedIn = processedOut = count`, but that must be _enforced_, not assumed.
3. **Settings.** Settled and shipped — device-side settings are read-only scalars; `DeviceBlockState` was deleted (see "Decisions worth keeping"). Do not resurrect a settings mirror.

### 18.2 Fix list (issues found — use-case A)

> **STALE (July 2026) — a record, not a work list.** Read the OPEN table in the READ FIRST block instead; every
> item here predates D5, the `Domain`→`SubGraph` rename and the history rebuild.

Ordered by whether they bite today.

- [ ] **F1 — auto-parallel silently discards device-side state.** The block is memcpy'd into USM and never copied
      back, so a mutated member or a `publishTag()` on the device copy vanishes; `mergedInputTag()` returns a view
      aliasing host tag-ring memory, so dereferencing it in a kernel is UB. Make it a compile error (no `mutable`
      members; no tag access on the auto-parallel path), not a silent loss.
- [x] **F2 — DONE** (verified 2026-08-21: resources keyed `(kind, deviceIndex)` with a per-queue USM resource, `SyclRuntime.hpp:68,87,121-124`). Original text: **multi-GPU USM is a context mismatch.** The USM provider ignores `deviceIndex`, so `gpu:sycl:1` yields a
      device-1 context but default-queue USM; SYCL USM pointers are context-bound. One `UsmMemoryResource` per
      registered queue, keyed like `SchedulerRegistry`.
- [x] **F3 — DONE** (`63a22ddf`) warn-once per block via `Block::markDeviceFallbackWarned()`.
- [ ] **F4 — silent failures**: `copyHostToDevice`/`copyDeviceToHost`/`glCtx.dispatch` still return `void`.
- [x] **F5 — DONE** (`63a22ddf`) the shader compiler's error string now travels into the `gr::Error`.
- [x] **F6 — DONE** (`63a22ddf`) dispatch uses `tryResolve`; an unserviceable domain says so and errors when the
      block has no CPU path. `dispatch` now returns `std::expected<work::Status, gr::Error>` (§18.1 decision 1).
- [x] **F7a — DONE** (`1d0324de`) seam gates on `GR_DEVICE_HAS_ANY_BACKEND`; GL detection moved to
      `BackendDetect.hpp` (no EGL headers in every TU); `gnuradio-core` links EGL/GL when the backend is enabled.
- [ ] **F7b — still open: nothing registers a `DeviceContextGLSL`.** Needs `registerGlslRuntime()` publishing a
      `gpu:glsl` domain (mind gotcha G18: `eglTerminate` invalidates the context — the test code leaks a Meyer's
      singleton on purpose). Until then `gpu:glsl` resolves to nothing and the seam warns + falls back to the CPU.
- [x] **F8 — DONE** (`124a13d0`) path chosen by the _resolved_ backend when a block offers both escape hatches;
      `FFT2`'s identity shader stub deleted (it would have returned the input unchanged).
- [x] **F9 — DONE** (verified 2026-08-21: `ShaderFragment.hpp:42` — "the element count is a uniform, not a baked literal"). Original text: **shader cache blow-up.** `count` is baked into the shader source, so every distinct work quantum
      compiles a program, retained forever in an unbounded map; `ShaderCache`'s LRU is never instantiated. Hoist
      `count` to a uniform, wire the LRU.
- [x] **F10 — DONE** (verified 2026-08-21: `schedulerCache` resolves once per block and is reused, `ExecutionStrategy.hpp:170-188`).
- [ ] **F11 — `ShaderFusion` is dead outside tests** (two callers, both `qa_DeviceContext`). Either a graph pass
      calls it or it goes.

### 18.6 Open questions for the author

1. **Fan-out** — all fan-out edges of one output port share a single buffer, so a host consumer and a device
   consumer on the same port cannot differ in residency. Accept, or split per-edge buffers?
2. **GLSL typing** — which stream types beyond `float` must the shader path carry (`gr::complex`? integers?).
3. **Fixed ratio** — enforce via a `Resampling<>` static check, or a runtime guard that errors when `count` is not
   the expected multiple?

(Answered since, and removed: the settings model — settled, `DeviceBlockState` deleted; residency — resolved by §D3
as VMM interior / USM boundary; use-case B's home — resolved by §D2, it rides the normal graph flow.)

## 19. Working practice (author's directive, 2026-07-10)

- **Opus orchestrates**: owns the design, reviews it, decides, and verifies. It does not delegate judgement.
- **Sonnet agents do the coding and bug-fixing**, and **never more than two at a time** — concurrent agents editing the
  same tree fight each other.
- Reserve the heavier design model for a genuinely new design fork; it was wrong four times on settled ground here
  (Settings move-vs-copy assign, `gr::Block<T>` as the base, `migrateField` compatibility, and an invented
  `processChunk` API).
- **Empirical probes beat analysis.** Every contested claim in this work was settled by running code — the RTX 3070
  mirror experiment, the `pmr::string` SSO probe, the `PmrMigratable` trait table, the `Tensor` ctor mismatch, and
  `parse("cpu:sycl") -> kind='host'`, which four rounds of analysis had missed.

## Current approach — device edge residency (AUTHORITATIVE; supersedes the deleted managed-sub-Graph appendices)

### D1 · why the managed sub-Graph was abandoned — root cause (2026-08-07)

> **⚠️ MECHANISM CORRECTED 2026-08-17 — the conclusion below is wrong about WHY, and it matters.**
> The exported weakRef does NOT merely establish topology: `applyEdgeConnection` creates the ring on the source
> port (`Graph.hpp:788`) and `sourcePort.connect(destinationPort)` (`:791` → `Port.hpp:851`) really does hand the
> member's input port a reader on the parent's buffer **through the weakRef alias**. The bridge is built.
> It is then **destroyed** by the inner scheduler's own start-up: `SchedulerBase::start()` calls
> `disconnectAllEdges()` unconditionally before `connectPendingEdges()` (`Scheduler.hpp:658-662`), and
> `Graph::disconnectAllEdges()` (`Graph.hpp:823-844`) is a **PORT SWEEP, not an edge sweep** — it walks every
> block's every dynamic port and calls `Port::disconnect()` (`Port.hpp:1034-1041`), which replaces `_ioHandler`
> wholesale. The parent's share is wiped; the boundary edges live in the PARENT's `_edges`, so the inner graph
> cannot re-apply them and the parent won't (already `Connected`). Hence `sink.count == 0`.
> **Consequences:** (1) the blocker is NOT inherent to port export; (2) a sub-graph scheduler that does not
> inherit `SchedulerBase` — writing its own `start()` that never sweeps — keeps the parent's boundary bindings
> **by construction**, which is exactly the H2D/D2H staging the device case wants; (3) the generic vehicle could
> be fixed for everyone by making `disconnectAllEdges()` edge-scoped (no signature change; UNVERIFIED whether
> anything relies on the sweep clearing out-of-edge-list connections — probe = apply + run `qa_Scheduler`,
> `qa_Graph`, `qa_ManagedSubGraph`, `qa_DynamicPort`). `EdgeState::Overridden` (`BlockModel.hpp:75`) exists,
> is used nowhere, and already makes `connectPendingEdges` skip an edge (`Graph.hpp:854`) — a ready-made skip mark.
> Note also a SECOND, distinct trap on the runtime-message wiring path: pending edges are applied only at
> `start()`/`resume()` (`Scheduler.hpp:662`, `:921`), never while RUNNING, so a probe that wires at runtime without
> a pause/resume shows `count == 0` for an unrelated reason. Distinguish by logging `edge.state()` after 3 s:
> `WaitingToBeConnected` ⇒ trap 2, `Connected` ⇒ trap 1.

**Finding:** a CPU probe — `SlowSource → SchedulerWrapper<scheduler::Simple>[Copy → Copy] → CountingSink`, wired via the
SAME message-based export/connect `qa_ManagedSubGraph`'s lifecycle test uses (export pass1.in as `inExp`, pass2.out as
`outExp`; emplaceEdge source.out→inExp, outExp→sink.in) — leaves **`sink.count == 0`** after 3 s. Stream data does NOT
reach the sink through the managed sub-Graph's exported ports. Pure CPU, `scheduler::Simple`, no device code ⇒ a property
of the vehicle, not the device work.

**Blocks:** the chosen vehicle `SchedulerWrapper<scheduler::Device>` relies on the exported boundary ports carrying stream
data (source → exported-in → first member; last member → exported-out → sink); scheduler::Device's boundary legs then
read/write those member ports. Structural-only export ⇒ those legs get no data ⇒ steps 3–5 blocked. The stalled step-3
agent hit the same wall independently.

**Named crack:** Fable's §3 cited `qa_ManagedSubGraph.cpp:170-173` for "parent edges terminate on the member's actual
port" — but that test only asserts STRUCTURE (edge/connection counts, lifecycle states), never end-to-end stream data.
That citation was never a data-flow proof; it is the unverified foundation the vehicle sat on.

**Caveat:** the probe used RUNTIME message-based export (the only proven in-tree wiring). A BUILD-TIME export (what
scheduler::Device would use) is untested — but the build-time agent attempt saw the same no-flow, so likely a genuine gap.

**Solid:** steps 1 (`4d56a610`) + 2 (`71b5c5b4`) are committed + fully gated (CPU byte-identical + device tests green),
reusable regardless of the vehicle. Tree clean; agent's scheduler WIP left untracked (`device/SchedulerDevice.hpp`,
`blocks/fourier/test/qa_SchedulerDevice.cpp`).

**Checkpointed to user** (batch-authority: pause on a blocking scope question; do NOT pivot the vehicle unilaterally).
Options: (A) fix the managed-subgraph export to bridge the parent edge's stream buffer to the member's port (core
Graph/SchedulerWrapper change; benefits managed subgraphs generally; the probe is its RED test); (B) a bespoke
device-region construct owning the boundary transfers directly, bypassing the export machinery (deviates from the
managed-sub-Graph choice); (C) root-cause first (build-time vs runtime export; the exact bridging step).

**ROOT CAUSE CONFIRMED — fundamental, NOT a timing issue (build-time tested, 2026-08-07).** A BUILD-TIME
export/connect (`wrapper->exportPort(...)` + `graph.emplaceEdge(source→inExp, outExp→sink)` BEFORE `exchange`) ALSO
yields `sink.count == 0`. The `emplaceEdge` connect SUCCEEDS (returns has_value; topology correct) — only the stream
buffer is not bridged. So it is not build-time-vs-runtime: the exported `DynamicPort` weakRef (`Graph.hpp:255`)
establishes TOPOLOGY only; the parent's edge buffer (source→inExp) and the member's port buffer (pass1.in, materialised
by the inner scheduler's own `connectPendingEdges`) are two separate buffers with no share/copy across the
managed-subgraph boundary ⇒ source data never enters the subgraph. **Fix must bridge the boundary buffer:** SHARE (the
exported port's buffer IS the member's port buffer — the inner scheduler must not separately materialise it) OR COPY at
the wrapper boundary (wrapper work() copies parent-edge↔member-port per iteration). For the device case the boundary
COPY _is_ the H2D/D2H — a boundary-copy fix doubles as the device transfer — but the parent's data must first reach a
buffer the wrapper can read. Probe code (runtime + build-time, both RED) preserved in this note for reuse as the fix's
RED tests. `emplaceEdge` sig = `(srcBlk, srcPort, dstBlk, dstPort, minBufferSize, weight, edgeName)` — 7 args.

### D2 · redirection — device transfer as a property of the edge (2026-08-07)

**State:** branch `syclExperiments` @ `71b5c5b4`; backup branch/tag `*-20260713`. Steps 1+2 COMMITTED + gated (device
kernel-body split `4d56a610`; `processBulkDevice` block entry point `71b5c5b4`; CPU byte-identical qa_PerformanceMonitor/
qa_ManagedSubGraph + acpp qa_DeviceBlockStyles all green). Tree clean. Untracked leftovers from the ABANDONED
scheduler::Device attempt: `core/include/gnuradio-4.0/device/SchedulerDevice.hpp`,
`blocks/fourier/test/qa_SchedulerDevice.cpp` — removable (superseded).

**APPROACH REDIRECTED (supersedes the old steps 3–6 / the managed sub-Graph vehicle).** The managed sub-Graph is BLOCKED:
it never bridges the stream buffer across the boundary (§D1 — fundamental, both build-time and runtime). NEW foundation
(user-agreed 2026-08-07): **device transfer = a property of the EDGE's DOMAIN CROSSING**, resolved in `Block<T>::work()`/
`ExecutionStrategy::dispatch` by buffer residency — implicit H2D/D2H at host↔device crossings, ELIDED at device→device
edges. One rule ⇒ implicit single-GPU-block AND on-device multi-GPU-block chaining, via the NORMAL graph/scheduler flow
(no managed sub-graph, no separate scheduler, no export buffer-bridge).

**Two pieces to build:**

1. _residency elision in dispatch (small; builds on step 1)._ `dispatchAutoParallel` already elides via
   `ctx.isDeviceAccessible(span.data())` (`ExecutionStrategy.hpp:402-432`); replicate for the framework-processBulk tier
   (`dispatchDeviceBulk`, reusing step-1 `runDeviceBulkCore`) and the sycl-hatch. ~15 lines each. Compile-gated ⇒ CPU
   path untouched.
2. _device-domain edge buffer WITHOUT the host wrap-mirror — THE OPEN DECISION._ A device-USM `CircularBuffer` still runs
   the host memcpy wrap-mirror at EVERY publish (`CircularBuffer.hpp:393-407`) — that is the ~8× tax, on the WRITE of each
   device→device edge. Options: **(a)** device-aware wrap-copy inside `CircularBuffer` (device `memcpy` at wrap;
   `UsmMemoryResource::queue()` exists — §6; user green-lit touching CircularBuffer as fallback); **(b)** a distinct
   device-edge buffer type. **USER TO DECIDE (a) vs (b) — this is the next question.**

**Next steps:** (i) decide piece 2 (a vs b); (ii) implement piece 1 (dispatch elision) + piece 2; (iii) validate a
2-GPU-block chain keeps data on-device across the device→device edge (residency test; §6 is the design); (iv) measure
vs standalone `bm_FFT_backends` per N — the recover-the-~8× goal. qa_PerformanceMonitor MUST stay byte-identical
throughout.

**Reusable:** the data-flow probe (`SlowSource → SchedulerWrapper<Simple>[Copy→Copy] → CountingSink`, assert
`sink.count > 0`) is the RED test IF managed sub-graphs are later fixed to SHARE the boundary buffer — a separate,
orthogonal effort, NOT on the device critical path.

### D3 · FINALISED design — device edge buffer (all decisions locked, 2026-08-07)

**Memory model (Re1):** host↔device BOUNDARY edges → USM (shared/managed, host-accessible, implicit migration; reuse the
existing `UsmMemoryResource`). device→device INTERIOR edges (a device-only domain) → a VMM double-map resource
(device-ONLY, no host touch, no wrap-mirror). Edge classification: BOTH endpoints device + same device ⇒ interior ⇒
VMM double-map; otherwise ⇒ crossing ⇒ USM.
**Capability query (= option B):** free function `gr::usesMMAP(std::pmr::memory_resource*)` over a thread-safe
function-local-static REGISTRY; each double-mapping resource registers itself on construction, deregisters on
destruction. NO inheritance from STL types (`gr::memory_resource : std::pmr::…` rejected).
**Hard compile constraints (Re2):** `CircularBuffer` + the registry + `usesMMAP` must compile **no-RTTI, no-exceptions,
and without any SYCL/CUDA header**. ⇒ registry + `usesMMAP` live in CORE; the device VMM resource lives in the device
layer (behind `GR_DEVICE_HAS_*`) and registers itself into the core registry. Use `std::expected`/error codes, never throw.
**Resource implementation (Re3):** REUSE the existing PMR wrapping of SYCL/USM alloc/dealloc (`UsmMemoryResource`, which
already exposes `queue()`); the VMM double-map resource follows the SAME PMR pattern + queue/context plumbing, adding
`cuMemCreate` + `cuMemAddressReserve(2N)` + `cuMemMap`(offset 0 and N) + `cuMemSetAccess` in `do_allocate`, and the
inverse in `do_deallocate`. Level-Zero / other backends: their own VMM or fall back.
**Granularity (Re4):** assume 2–4 MB minimal device-memory overhead (VMM allocation granularity) — a per-edge floor,
ACCEPTED. ALSO apply the same large-page (2–4 MB) granularity to the USM BOUNDARY edges (shown beneficial for H2D/D2H
transfer throughput).
**CircularBuffer touches (ALL internal; public API + wrap logic untouched):** (i) line 894 identity check →
`gr::usesMMAP(allocator.resource())`; (ii) line 299 `align_with_page_size` → resource-aware granularity (not host
`getpagesize()`); (iii) line 893 — un-gate the query from `has_posix_mmap_interface` (host POSIX-mmap ≠ device VMM).
**Verify-first STEP 0 (before any resource code):** a standalone VMM probe — (a) does `sycl::get_pointer_type` report a
`cuMemMap` range as DEVICE (must, so piece-1 `isDeviceAccessible` elision composes); (b) `cuMemGetAllocationGranularity`;
(c) a real device-USM `CircularBuffer` NEVER host-derefs the buffer for trivially-copyable T (no host default-construct;
mirror gated). If (c) fails → also gate the construction.
**Two pieces, both required:** (1) dispatch RESIDENCY-ELISION (device→device edges run in place, no H2D/D2H — extend
`dispatchDeviceBulk` + sycl-hatch like `dispatchAutoParallel` already does via `isDeviceAccessible`); (2) THIS device
edge buffer (VMM double-map interior / USM boundary, no wrap tax). Neither alone gives on-device chaining.
**Build order:** step 0 probe → device VMM resource (+ register) + the CircularBuffer touches → wire ComputeRegistry so
device-interior edges resolve to it → piece 1 elision → validate a 2-GPU-block chain stays on-device → measure vs
standalone `bm_FFT_backends`. qa_PerformanceMonitor MUST stay byte-identical throughout. Recommend a Fable pressure-test
of this §D3 before the resource code.

### STEP-0 RESULT — all three sub-probes PASS on hardware (2026-08-07, RTX 3070 / acpp generic / driver 610.57.04)

Probe sources: `vmm_probe.cpp` (a+b, acpp + CUDA driver API) and `touch_probe.cpp` (c, gcc15, no GPU) — kept in the
session scratchpad. §D3 is now empirically verified, not merely reasoned. **Nothing in §D3 needs to change.**

**(a) `sycl::get_pointer_type` reports a `cuMemMap` range as `device` — at BOTH mappings (offset 0 and offset N).**
So `DeviceContextSycl::isDeviceAccessible` (which is literally `get_pointer_type(ptr) ∈ {shared, device}`,
`DeviceContextSycl.hpp:68`) returns TRUE for VMM memory ⇒ **the piece-1 residency elision composes as designed**. This
was the plan's #1 risk: had it returned `unknown`, dispatch would have allocated a host shadow and `copyHostToDevice`
_from_ device-only memory, and residency would have had to come from registry metadata instead. It does not.

- **(a2) bonus, also proven:** a kernel written through mapping #1 is read back correctly through mapping #2
  (`base[0..8) = 100+i` → `base[N..N+8)` reads `100..107`). The double map genuinely aliases on-device ⇒ **the ring's
  wrap needs no mirror at all on this path**, which is the whole point of the VMM resource.
- **Context caveat worth keeping:** acpp leaves NO CUDA context current on the calling thread, yet the kernel reached
  memory mapped in the **primary** context ⇒ acpp uses the primary context. The VMM resource must therefore map into
  `cuDevicePrimaryCtxRetain`'s context (a `cuCtxCreate` of its own would fault: VMM mappings are per-context).

**(b) granularity = 2 MiB exactly — `CU_MEM_ALLOC_GRANULARITY_MINIMUM` == `..._RECOMMENDED` == 2097152.**
Confirms Re4's "assume 2–4 MB per-edge floor" precisely; no need to hedge higher. Read it from the driver at runtime
rather than hard-coding, but 2 MiB is the number on GA104.

**(c) the host-touch list is EXACTLY 2 dereference sites — there is no hidden 5th touch.** Configuration probed:
`CircularBuffer<float, dynamic_extent, ProducerType::Single, SleepingWaitStrategy>`, i.e. trivially-copyable `T` — which
is the scope of the phase-4 result and the reason for the `is_trivially_copyable_v` caveat in step 2 below. Method
(stronger than a code read): hand `CircularBuffer` a PMR resource whose region is `mmap`'d `PROT_NONE`, install a SIGSEGV handler that
records the faulting instruction, un-protects that one page and retries, and **re-protect the whole region between
phases and on every round** (without that, phase 1 un-protects everything and every later phase reports a false zero —
the first version of the probe did exactly that). Every host touch then reports itself with an address.

| phase                                                                     | faults                     | attributed to                                                                                 |
| ------------------------------------------------------------------------- | -------------------------- | --------------------------------------------------------------------------------------------- |
| 1 construction                                                            | 2 (1 site)                 | `std::construct_at` ← the element-construct loop, `CircularBuffer.hpp:899-917`                |
| 2 control (probe writes payload deliberately)                             | 2                          | `main` + mirror — proves the handler is live, not silently disarmed                           |
| 3 **200 rounds publish/consume across many wraps, probe touches nothing** | 400, **all** libc `memcpy` | **one single caller: `WriterSpan::~WriterSpan()` `CircularBuffer.hpp:401`** = the wrap mirror |
| 4 destruction                                                             | **0**                      | the deleter's `destroy` loop is empirically a **no-op** for trivially-copyable T              |

Attribution used the return address at `[RSP]` (libc's AVX `memcpy` is a leaf, so `[RSP]` is still its caller) — all
401 memcpy faults resolve to that one `~WriterSpan` line. The only non-libc sites in the whole run are the probe's own
`main` and the construct loop. **Therefore: the reader path, the claim strategy, and destruction never dereference the
data region at all.** The two derefs are the two §D3 already names (i, iv); (ii) granularity and (iii) un-gating are
sizing/control-flow, not derefs. The 4-touch plan is complete and the `{usesMMAP, deviceOnly}` two-flag split is
sufficient — skipping the mirror + the construct loop leaves nothing else to gate.

**PLAN CORRECTION found while auditing step 0 (amends §D3 + step 2 — the doc previously read as settled and was not):**
the "skip the element-construct loop if `deviceOnly`" rule must NOT be unconditional. For a non-trivially-copyable `T`
skipping construction is UB on every later access, and the deleter's `destroy` loop — a no-op for trivial `T` only
(phase 4) — would then run real destructors over never-constructed objects. The buffer does contemplate non-trivial `T`
(the mirror's `else` branch uses `std::copy_n`). Pick one, before writing step 2: gate the skip as
`deviceOnly && std::is_trivially_copyable_v<T>`, **or** reject the pairing loudly (a device-only resource meeting a
non-trivial `T` ⇒ `static_assert`/allocation failure). Preference: **reject loudly** — device edges carry
kernel-transferable types anyway, and a hard failure beats silent UB.

**ALSO found while auditing (a genuine PRE-EXISTING bug, unrelated to the device work, cherry-pickable to `main`):**
`double_mapped_memory_resource` **leaks its whole upper mapping on every buffer destruction.** `do_allocate(N)` maps
`2*N` of VA (`:129`, upper half aliasing the lower) but `do_deallocate(p, size)` calls `munmap(p, size)` — only `N`
(`:151`). Measured on the DEFAULT host path, 150 create/destroy cycles of a 1 MiB logical ring: **VA +153600 KiB,
`/proc/self/maps` entries +150, and system `Shmem` +153584 KiB — so real RAM, not just address space.** The backing
store is a `memfd`, not a POSIX shm object, and its fd is closed right after mapping — so the orphaned mapping is the
only thing keeping the memfd's pages alive, and process RSS stays flat while system `Shmem` grows (look at `Shmem`, not
RSS). It scales with buffer churn — scheduler restart/reset and port resize both destroy buffers — and the mapping-count
growth also walks toward `vm.max_map_count` (~65530).
**FIXED in the working tree (uncommitted, awaiting authorisation):** `munmap(p, 2 * size)` plus a regression test in
`qa_buffer.cpp`'s `DoubleMappedAllocatorTests` ("deallocate releases both mappings", Linux-gated, asserts the process's
virtual-page count does not grow over 64 alloc/dealloc rounds). Proven RED→GREEN **in-tree**: without the fix the test
reports `virtual pages grew 6528 -> 10624` (exactly the predicted 4096-page leak) and the binary exits 255; with it,
`qa_buffer` + `qa_Port` + `qa_Scheduler` + `qa_PerformanceMonitor` all pass. Probe: `leak_probe.cpp`. Keep it a
standalone commit — it is independent of the device work and cherry-pickable to `main`.

## Follow-up (2026-08-13): sample_rate rescale vs. DataSet-producing blocks

The spectrum FFT (`blocks/fourier/.../fft.hpp`) now opts out of the framework's resampling
`sample_rate` rescale (Settings.hpp:1198) by erasing `sample_rate` from `forwardSettings` in its
3-arg `settingsChanged` — the reflected field keeps the _input_ rate, so the frequency axis is
physical (Nyquist = fs/2). TODO: audit the other DataSet-producing resampling blocks (e.g.
StreamToDataSet, spectral estimators) for the same axis/metadata corruption and apply the same
opt-out where `sample_rate` is input-signal metadata rather than an output stream rate.

## 20. WebGPU/GLSL cascade — design record (2026-08-16). BUILT, but by a different vehicle: see §22 and §29.

> **HISTORICAL — this section is cited elsewhere as an entry point; it is no longer one.** The cascade was
> built, measured, and then removed with the shader backends under D5. Read §50 and §55 for what replaced it,
> and the READ FIRST reading key for the renames. Everything below is a record.

### 20.1 The instruction and the two hard requirements

Maintainer: _"replace the per-block dispatch with the plan/cascade model. Evaluate first whether the non-host/non-SYCL
ports could be enhanced by having an optional domain-specific handler that keeps the buffer references/handles (even if
not pointers) to mimic the cyclic buffer behaviour … a common shared working space/copy that is passed through each
blocks GLSL/WebGPU processing function would achieve the same result."_

Two named requirements, which are **different problems with different answers**:

- **(a) RAW ordering** — in `host→A→B→C→host`, B must not get the handle before A has finished with it.
- **(b) WAR / anti-dependency** — in `host→A→B→host` + `A→C→host`, B must not modify A's output before C has an
  unmodified copy. **This is not a barrier problem**; a barrier makes C wait, not makes it correct.

### 20.2 Vehicles evaluated — ranking **V3 > V1 > V2** — HISTORICAL: NONE of these was built

> The vehicle actually chosen is a FOURTH one (§22): the managed sub-Graph for shader backends, with SYCL keeping
> both models. V3's ring-keyed mirror was never built — §29 replaced it with the aside channel, which needed no ring
> identity at all. Kept only for why V1 and V2 were rejected; V2's Port.hpp:551 proof is still the reason token
> edges are a dead end.
> Ground truth: pre-branch (merge-base `f60d1a69`) already has `ComputeDomain`/`Access` (ComputeDomain.hpp:16-18) and
> `Edge` with `_domain`/`_dataResource` (BlockModel.hpp:74,92-95). The `device/` tree, `InputViewLike`/`OutputViewLike`
> (Buffer.hpp:76-79) and all dispatch tiers are branch-only.

**V3 — shadow / aside channel. RANKED FIRST. This is the maintainer's "optional domain-specific handler" made concrete.**
The edge keeps its ordinary `CircularBuffer<T>` and all its accounting; it gains _alongside_ it a device mirror
(`WGPUBuffer` / SSBO) **keyed on ring identity**. Interior consumers read the mirror and skip the host round trip; the
host ring is materialised only at a device→host boundary or on CPU fallback.

- pre-branch cost ≈ **45 lines, all additive**: (a) a stable ring-identity accessor on `ReaderSpan`/`WriterSpan`
  (CircularBuffer.hpp:632-758 / 338-457) ~15 lines — _this is exactly the gap the branch already documents twice_,
  at ExecutionStrategy.hpp:338-343 and DeviceContextGLSL.hpp:42-53; (b) one optional handler member on `Edge`
  (BlockModel.hpp:74) ~5 lines; (c) extend the `Access::DeviceOnly` upgrade (Graph.hpp:734) with the existing
  `hasSameSourcePort` scan pattern (Graph.hpp:802-821) + one `BlockModel` virtual exposing device capability at
  connect, ~25 lines.
- `graph.connect` UX **fully survives**; blocks stay ordinary scheduler blocks.
- `work()` contract **holds unchanged** — hatches still consume/publish themselves (Block.hpp:1957-1977;
  fft.hpp:276-277); the elision lives inside the dispatch tier.
- tags/accounting **untouched** — the tag axis is already a separate always-host ring (Graph.hpp:748-751).
- ≈ **400-700 branch lines**. Riskiest piece: **ring wrap** (see 20.7).
- **Serves GLSL identically** (fixes the per-block-mirror split at DeviceContextGLSL.hpp:42-53 and the
  upload/download-every-call at ExecutionStrategy.hpp:344-352) and **leaves SYCL untouched** (SYCL keys on
  `isDeviceAccessible(span.data())`, ExecutionStrategy.hpp:476-477, 569-570, and never consults the aside handler;
  on real-USM edges the slot stays empty).

**V1 — composite plan-executor block. RANKED SECOND.** One scheduler block, one host in-edge, one host out-edge; interior
members are plan entries, not scheduler blocks. **Zero pre-branch change**, and the PoC already proves the shape. But
interior `graph.connect` UX is lost by construction, interior entries cannot publish tags, and every kernel must be
re-encoded into plan ops or grow a second handle-based hatch signature (today's `processBulk_webgpu/_glsl` take host
spans and do their own upload/readback, fft.hpp:224-279). ≈500-800 lines. _Good product vehicle, poor framework answer._

**V2 — token-carrying device edge. DEAD END, do not build.** Proof line **Port.hpp:551**: `BufferType` is a
_compile-time port attribute_, `ReaderType/WriterType/IoType` are fixed from it (Port.hpp:564-566), `resizeBuffer` swaps
only the pmr resource (Port.hpp:925-933), and `Port::connect` static-asserts `value_type` equality (Port.hpp:1044-1048).
A token ring can never be decided **per edge** at connect. Worse: `work()` accounting is in sample units end-to-end
(Block.hpp:2074-2100, 2113-2150, 2211-2212) and tags map via `reader.position()` in ring items (Port.hpp:602-605), so a
full sample↔token translation layer through Block.hpp is new machinery, not preservation. And host fallback becomes
fatal rather than graceful: availability resolves _per work() call_, after connect (ExecutionStrategy.hpp:296-301), so
`dispatchCpuFallback` (ExecutionStrategy.hpp:606) would find no samples. ≈1000+ lines across three pre-branch types.

### 20.3 Backend synchronisation — spec-grounded, decides both requirements

- **(a) is FREE in WebGPU.** The spec has _no barriers at all_ (the word never appears). `GPUQueue.submit()` is defined
  as "for each commandBuffer … execute each command in order"; each dispatch is its own **usage scope** ("In a compute
  pass, each dispatch command … is one usage scope"), so the browser inserts whatever native barriers make one scope's
  writes final before the next runs. Ordering **and** visibility, within a pass, across passes, and across submits on
  the one FIFO queue. → https://gpuweb.github.io/gpuweb/#dom-gpuqueue-submit ,
  https://gpuweb.github.io/gpuweb/#programming-model-synchronization
- Producer and consumer of the **same buffer may share one compute pass**, provided they are separate ordered
  dispatches — usage-scope validation is _per dispatch_ in a compute pass (unlike render passes, which validate
  pass-wide). Read+write of the same buffer inside _one_ dispatch's own bind groups is rejected.
- **OpenGL is the opposite: `glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)` is MANDATORY** between two
  `glDispatchCompute` calls touching the same SSBO — explicitly not implicit. → https://docs.gl/es3/glMemoryBarrier
- **(b) has no runtime fix.** WebGPU offers no aliasing/versioning/snapshot mechanism; once an in-place write is
  encoded, no later consumer can see the pre-write value, and overlapping `copyBufferToBuffer` is a hard error. The
  spec's own answer is _allocate a separate destination_.
  → **Therefore the rule is: a mirror with more than one consumer is NEVER written in place.** In V3 this is
  _structural_, not policed: each block writes its **own output ring's** mirror, never its input's. In-place stays an
  optimisation to be applied only where a value has exactly one consumer and this is its last use — and the
  connect-time all-consumer scan (20.2c) already computes the fan-out that gates it.

### 20.4-20.6 — SUPERSEDED, DELETED

These were: a Phase-0 probe to run first, a phased deliverable, and three blocking questions to re-ask. All are
answered. **Q1 (ring identity) was MOOT** (§21.1). **Q2 (fill policy) = skip + a debug flag**, shipped as
`debug_fill_host_rings` (§33/`51dac107`). **Q3 (first backend) = both**, and both now work. The **Phase-0 probe was
never needed**: a domain drives its members itself, so one-encoder/one-submit is available by construction, not as a
conditional phase. The vehicle is the Domain (§22), and the elision is the aside channel (§29), not the phased
mirror planned here.

### 20.7 Verify, do not assume (advisor-flagged)

- **The ordering invariant is load-bearing and unchecked.** The (a)-is-free argument assumes A's `wgpuQueueSubmit`
  happens-before A's ring publish, and B's encoding happens-after B observes it. **Confirm the dispatch tier submits
  before `publishSamples`/`consumeReaders`**, and state it in a comment at the mirror-validity site. Moot in-browser
  (single-threaded) but the same code serves GLSL on native under multi-threaded `gr::scheduler::Simple`.
- **Ring wrap is the top risk** and produces subtly wrong numerics rather than clean failures: a `WriterSpan` that wraps
  writes two disjoint regions, and the consumer's `ReaderSpan` may wrap at a _different_ offset, so a high-water mark
  over a linear index does not by itself say which mirror bytes are valid. **Write the wrap test FIRST**, with a ring
  size chosen so the chunk straddles the boundary.
- `minStorageBufferOffsetAlignment` is a **queried device limit**, not a hardcoded 256 (the 256 figure was UNVERIFIED).
- V3's skip-fill hazard is **not fully connect-time decidable**: static capability is (`canDispatch`,
  ExecutionStrategy.hpp:162-168) but CPU fallback is per-work-call (ExecutionStrategy.hpp:184-187, 235-241, 299-301).
  Sound protocol = skip-fill only when the connect-time all-consumer scan passes; runtime fallback materialises via the
  handle. Plain blocks with no `compute_domain` never enter dispatch (Block.hpp:1939-1947, 1988).
- UNVERIFIED: whether Asyncify tolerates V3's flush points _mid-_`work()`. The PoC proves only the V1 single-boundary shape.

### 20.8 What the PoC does and does NOT de-risk (`webgpu_mvp/main.cpp`)

Does: `DevicePlanSnapshot` as flat `vector<int>` ops (4 ints/op) + flat `vector<float>` params (191-198, 261-276);
rebuild gated on a `kind:paramCount` signature (467-477, which notably excludes versions and param values); **one**
encoder, **one** compute pass, **one** submit for the whole chain (569-587), with no barrier call anywhere.
**Does NOT: fan-out — at all.** `streamBuffers[i]→[i+1]` (556-563) is a strict linear chain; `HostPipeline._blocks` is a
flat `vector<BlockRecord>` with no graph/edge structure; there is **no aliasing, liveness, refcount or copy-insertion
logic in the file**. Also no in-place anywhere: every index is a distinct `GPUBuffer` (529-536), fresh per intermediate
edge — not pooled, not ping-pong. **Requirement (b) is therefore entirely new work that the PoC does not de-risk.**

## 21. Re-analysis under the four maintainer requirements (2026-08-17)

Requirements restated by the maintainer: R1 same C++ DSP function on host or any backend · R2 a CPU/SYCL-only user
pays nothing · R3 `CircularBuffer`+PMR stays first-class, device handle subordinate · R4 clean over compatible
(no device-path users yet).

### 21.1 Q1 (ring identity) is MOOT — no pre-branch widening needed

The premise was "a span exposes no path to its ring". True, but irrelevant: `ExecutionStrategy::dispatch` receives
the BLOCK. `Port::streamReader()`/`streamWriter()` (`Port.hpp:979-997`) and `Port::buffer()` (`:960`) are public, as
are `Reader::buffer()`/`Writer::buffer()` (`CircularBuffer.hpp:826`, `:519`). Identity is reachable at `92278b62`
with zero core change. Caveats: `buffer()` copies a `shared_ptr` (churn — `CircularBuffer.hpp:520` documents avoiding
it), so cache identity per connect/settings-epoch, not per `work()`; `Port::resizeBuffer` (`:925-958`) replaces the
ring, so identity legitimately changes there and must invalidate the mirror.

### 21.2 R2 is achievable at literally zero — and it dissolves decision #5 rather than reversing it

Target measured/derived: `sizeof(Block<Copy>)` back to main's exactly (the 40 B = `_settingsEpoch` 8 +
`_deviceShadow` 24 + `_deviceScheduler` 8 all DELETED, not relocated), vtable back to main's, hot path back to
main's shape. Key move: **freshness-by-construction** — re-upload the block image per dispatch through the
strategy's existing per-call mirror path, which removes the reason the epoch and shadow existed at all. The
`__ACPP__` ODR hazard disappears because no member is ever conditional: layout is main's layout in every
configuration. Device headers move to an optional module gated by a CMake-declared `GR_DEVICE_ENABLED`.
Residual unconditional core cost ≈ +190 preprocessed lines (~0.07% of a 282k-line TU), all backend-free.

### 21.3 R1 — the honest limit, stated once

SYCL-class and host scalar/SIMD: literally one `processOne`/`processBulk` body. Shader backends: GLSL/WGSL consume
TEXT, so a second surface is irreducible for FFT-class (multi-stage dispatch, workgroup memory, barriers) — the
per-backend kernels live BELOW the authoring line in the algorithm layer, and a block author adds zero lines for a
new backend. Map-class blocks could reach literally one body via a capture emitter (symbolic proxy through the
SIMD-generic `processOne`); that is a maintainer decision, not a prerequisite. Author surfaces collapse 8 → 3
(host `processBulk`, `processOne`, one `processBulk_device`) + interim `shaderFragment()`.

### 21.4 The device-region vehicle (V4) — what the code says

Injecting buffers and calling the member's own `processBulk` is NOT "just call the function". `Block::work()`
services split three ways: genuinely per-region (sample negotiation, EOS/epilogue flush, tag forwarding,
back-pressure) — the real saving; but PER-MEMBER obligations a direct call silently drops: lifecycle `start/stop`
hooks (e.g. `SignalGenerator::start()` configures `_core`), `applyChangedSettings()` (the FFT resizes chunk sizes in
`settingsChanged`; skipping it freezes `_settingsEpoch` so the device mirror serves stale settings forever),
consume/publish rate adaptation, error surfacing, and the scheduler-owned message pump. Worst: `mergedInputTag()` and
`publishTag()` are valid ONLY during dispatch (`Block.hpp:799`, `:1435-1453`) — outside it they silently return empty
/ write into a ring nobody reads. So the region wrapper becomes a mini-`workInternal`, not a thin caller.
`Edge::_dataResource` suffices for SYCL (device-domain edges already auto-resolve to registry USM,
`Graph.hpp:734-736,753-776`) but NOT for GLSL/WebGPU: an SSBO/`WGPUBuffer` is an opaque handle, not a
host-dereferenceable PMR arena — exactly the R3 split.

### 21.5 Managed sub-Graph + its own dispatch scheduler — VIABLE, with one condition

See the D1 correction above. The blocker was `SchedulerBase::start()`'s unconditional port-sweeping
`disconnectAllEdges()`, not port export. **Condition: the inner dispatch scheduler must not inherit `SchedulerBase`**
(its `init/start/stop/pause/resume/reset` are reserved, `Scheduler.hpp:95-105`; `customStart` fires AFTER the wipe).
A standalone `Block<GlslDispatch>` with `blockCategory = ScheduledBlockGroup` needs only the small
`SchedulerWrapper` contract and writes its own `start()` — parent boundary rings survive as H2D/D2H staging,
interior edges never get host rings (the parent descends into `TransparentBlockGroup` children only,
`Scheduler.hpp:586,610`). No `Graph` hook required for the device case.
**Open conflicts:** (i) `compute_domain` is a per-block SETTING (`Block.hpp:713`) but a region is TOPOLOGY — either
the graph auto-forms regions from maximal same-domain connected subgraphs at connect (the predicate already exists,
`Graph.hpp:734-736`) or the user hand-builds regions and R1 weakens; (ii) `SchedulerWrapper` runs its inner scheduler
on its own `std::thread` (`SchedulerModel.hpp` `_schedulerThread`), which fights single-threaded GPU submission and
Emscripten/Asyncify — a dispatch scheduler likely wants to be driven synchronously from the parent's `work()`;
(iii) SYCL would then have two execution models (per-block dispatch AND region scheduler) — decide or unify.
**Cheapest probe (~80 lines):** minimal standalone `PassthroughDispatchScheduler` inside `SchedulerWrapper`, wired
exactly as the D1 probe, assert `sink.count > 0`. Settles the whole vehicle before any device code is written.

### 21.6 MEASURED — FFT flat vs. managed sub-Graph (2026-08-17, gcc15 Release, `build-gcc15-release`)

Benchmark: `blocks/fourier/benchmarks/bm_fft_subgraph.cpp` (new, committed with the fix in `b0fcf8e0`). `CountingSource<float>` →
`FFT<float, DataSet<float>>` (fft_size 1024) → `CountingSink<DataSet<float>>`, free-running source, timed to 1024
DataSets (1 Mi samples) then stopped — steady-state throughput, so EOS/termination differences stay out of the number.
Sub-graph case = the FFT alone inside `SchedulerWrapper<Simple<multiThreaded>>`, ports exported as `inExp`/`outExp`,
parent attached via `emplaceEdge`. Best of 3.

| case                     | time    | throughput  | per sample   | DataSets |
| ------------------------ | ------- | ----------- | ------------ | -------- |
| FFT only (flat graph)    | 28.2 ms | 37.21 MS/s  | 26.87 ns     | 1024     |
| FFT in managed sub-Graph | 31.1 ms | 33.70 MS/s  | 29.68 ns     | 1024     |
| **sub-Graph overhead**   |         | **+10.4 %** | **+2.80 ns** |          |

**The vehicle now works, and it required the D1 fix.** A/B, same binary, only `Graph::disconnectAllEdges` differing:

- WITHOUT the fix: `dataSets=0`, every repetition burns the full 20 s deadline — D1 reproduced exactly.
- WITH the fix: `dataSets=1024` in 31.1 ms.
  This is the RED→GREEN proof that the blocker is the start-time PORT SWEEP, not the exported-port weakRef (§D1 note).

**The fix** (committed `b0fcf8e0`, `core/include/gnuradio-4.0/Graph.hpp`, +10/−13): `disconnectAllEdges()` becomes edge-scoped —
it still calls `initDynamicPorts()` on every block, then disconnects only the ports named by this graph's `_edges`
instead of sweeping every port of every block. A port not named by `_edges` may have been bound by a PARENT graph
across an exported sub-graph boundary; sweeping it destroys a binding neither graph will restore.

**Caveats on the number (do not over-read it):** +10.4 % is the cost of the sub-graph vehicle for ONE member block on
the CPU — it is dominated by the extra thread hop and the second scheduler's work loop, not by anything device-related,
and a region amortises that over N members. Both cases use `multiThreaded`; the source is free-running so the figure is
throughput, not latency. `sink.count` is polled across threads (benign torn read, one 200 µs poll of accuracy).
Not yet measured: N-member regions, single-threaded policy, and the `disconnect_on_done=false` interaction.

**Regression gate for the fix (gcc15 Release):** `qa_DynamicPort` · `qa_Block` · `qa_Graph` · `qa_ManagedSubGraph` ·
`qa_Scheduler` — **5/5 pass**. That covers the UNVERIFIED risk flagged earlier (something depending on the port sweep
clearing connections not named by `_edges`); nothing in these suites did. Wider ctest and clang20 not yet run.

### 21.7 VERIFICATION SWEEP for `b0fcf8e0` + `3a1f96a9` (2026-08-17) — GPU / AdaptiveCpp / browser

| lane                                                                             | result                                                                                                     |
| -------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| `bm_fft_subgraph`, gcc15 Release                                                 | flat 28.2 ms / 37.21 MS/s · sub 31.1 ms / 33.70 MS/s · **+10.4 %** · dataSets 1024                         |
| `bm_fft_subgraph`, **AdaptiveCpp** Release                                       | flat 26.7 ms / 39.22 MS/s · sub 27.9 ms / 37.62 MS/s · **+4.2 %** · dataSets 1024                          |
| `bm_FFT_backends`, **GPU RTX 3070** via acpp                                     | **105 PASS / 0 FAIL** across SimdFFT · SyclFFT:CPU · SYCL:CPU · SYCL:GPU · GLSL:GPU                        |
| acpp device suite, **GPU visible**                                               | **8/8 pass** (Context, Scheduler, BlockStyles, Seam, Residency, AutoParallel, ErrorChannel, LoggerBackend) |
| acpp device suite, **GPU hidden** (`ACPP_VISIBILITY_MASK=omp`, the CI condition) | **7 pass + 1 skip** (ErrorChannel skips by design without a real device)                                   |
| browser WASM+WebGPU, `tools/webgpu/fft_graph`                                    | **PASS, rc=0** — emcc build, headless Chrome, `GR4_FFT_GRAPH_RESULT: PASS`                                 |

GPU throughput sample, N=4096 (GFLOP/s-equivalent column of `bm_FFT_backends`):

| batch | SimdFFT | SyclFFT:CPU | SYCL:CPU | **SYCL:GPU** | GLSL:GPU |
| ----- | ------- | ----------- | -------- | ------------ | -------- |
| x1    | 16.5G   | 11.8G       | 1.3G     | 4.1G         | 1.6G     |
| x16   | 12.7G   | 11.7G       | 6.7G     | **41.0G**    | 15.7G    |
| x128  | 15.3G   | 11.4G       | 10.5G    | **76.3G**    | 14.0G    |

Reads as expected: the GPU only wins once the batch amortises launch+transfer; at x1 the CPU SimdFFT is 4x faster
than SYCL:GPU. Nothing here changes the "ship as capability, not performance" verdict.

**The sub-Graph overhead is TOOLCHAIN-DEPENDENT: +10.4 % (gcc15) vs +4.2 % (acpp).** Do not quote a single number —
the honest statement is "4-10 % for a single member block, amortised over N members in a real region".

**Browser caveat (unchanged environment limit):** the adapter is SwiftShader. `--use-angle=vulkan`,
`--enable-unsafe-webgpu` and `--enable-features=Vulkan` do NOT get headless Chrome onto the RTX 3070 here, so
`--require-hardware` still fails with exit 7 — the REJECTION path is proven, the ACCEPTANCE path remains UNVERIFIED.
The WebGPU numbers above are therefore correctness-only, never performance.

**Harness usability gap (follow-up, not blocking):** running the GR4 target rather than the PoC needs BOTH
`--js-name gr4_wasm_fft_graph.js` (the target emits a `_fallback.js` too, and the auto-detect refuses two candidates)
and `--expect "GR4_FFT_GRAPH_RESULT: PASS"` (the built-in `--expect` defaults are PoC-specific, so the run otherwise
exits 6 having actually PASSED). Worth defaulting per-target.

### 21.8 CORRECTION (2026-08-17) — the "+10.4 % gcc15 vs +4.2 % acpp" split was NOISE

Both earlier figures were single cold samples. Re-measured n=6 per compiler, both binaries correctly built from the
fixed source (the first gcc15 sample came from a binary that had NOT been rebuilt after `git apply` restored the fix,
which is also why an intermediate run showed `dataSets=0` — stale binary, not product flakiness):

| compiler            | flat (mean ± sd) | sub (mean ± sd) | overhead on means | overhead on minima |
| ------------------- | ---------------- | --------------- | ----------------- | ------------------ |
| gcc15 Release       | 25.00 ± 0.17 ms  | 26.00 ± 0.38 ms | **+4.0 %**        | +2.0 %             |
| AdaptiveCpp Release | 27.00 ± 0.73 ms  | 27.43 ± 0.75 ms | **+1.6 %**        | +2.3 %             |

**Sub-Graph overhead is ~2-4 % and is NOT systematically compiler-dependent** — on minima the two agree (+2.0 vs
+2.3 %), and the mean gap is within ~1 sd of the acpp spread. Quote **"a few percent for a single member block"**.
What IS systematic (sd 0.17-0.75, well separated) is absolute throughput: **gcc15 is ~8 % faster than acpp**
(25.0 vs 27.0 ms flat) — pure codegen (gcc15 `-O2 -march=native` vs the acpp/clang driver), unrelated to sub-Graphs.

Since the fix makes the boundary buffer SHARED rather than copied, the residual few percent is not data movement: it
is the second scheduler's thread and its work loop. A dispatch-style inner scheduler (§21.5) that does not run a
polling loop should shrink it further, but at 2-4 % for one member it is already near the measurement floor.

**`b0fcf8e0`'s commit message says "about +10%" — that number is superseded by this section.** Amend before pushing.

### 21.9 WebGPU hardware acceptance path — VERIFIED (2026-08-17), and the WebGPU perf gap

**The "no hardware adapter is reachable from headless Chrome here" note is now obsolete as a limitation.** The cause
was `--headless=new`, hardcoded in `run_wasm_webgpu.py` ahead of the pass-through args, so no `--chrome-arg` could
undo it. This box has a live session (`DISPLAY=:0`, `WAYLAND_DISPLAY=wayland-0`) and Vulkan sees the RTX 3070.
Added `--headed` to the harness (working tree, UNCOMMITTED): Chrome then runs on the real display and
`tools/webgpu/fft_graph` reports **`adapter: vendor=nvidia architecture=ampere`** with `--require-hardware`
**PASS, rc=0**. So the ACCEPTANCE path is verified, not just the rejection path.

**OBSOLETE — WebGPU HAS been benchmarked since `82db40cb`; see §21.10 and the idle-card reproduction in §35.**
The three bullets below describe the state before that and are kept only as the reason the benchmark was added:

- `bm_FFT_backends` has **no** WebGPU/WGSL backend (grep is empty) — it covers SimdFFT, SyclFFT:CPU, SYCL:CPU,
  SYCL:GPU, GLSL:GPU only, and predates the WebGPU work.
- `tools/webgpu/fft_graph/main.cpp` is a correctness/readiness harness — it negotiates the adapter and prints
  `GR4_FFT_GRAPH_RESULT: PASS`; it contains no timing (no `chrono`, no throughput reporting).
- `WgslFFT.hpp` is referenced only by `fft.hpp`'s hatch; nothing benchmarks it.
  So WebGPU is the ONE backend with zero perf data while the other five have a full N-sweep. Closing that needs either
  (a) timing inside `fft_graph` (small, in-browser, uses the block path end to end), or (b) a WebGPU backend row in
  `bm_FFT_backends` under an emscripten build (comparable to the existing table, but that benchmark is native-built
  today, so it is the larger change). Decide the vehicle before building.

### 21.10 WebGPU IS NOW BENCHMARKED (commit `82db40cb`, 2026-08-17)

`bm_FFT_backends` gains a WebGPU row via the existing `availableBackends()` seam — driver, sizes, batch counts,
repetition count and 5·N·log₂N convention all unchanged, so the row is directly comparable. Run it with
`tools/webgpu/run_fft_benchmark.sh` (bash; it delegates only the CDP driving to the existing Python runner, since
capturing console output needs a WebSocket client).

**Browser, WASM -O3, hardware RTX 3070 (`vendor=nvidia architecture=ampere`), ops/s:**

| batch | backend     | 1024  | 2048 | 4096 | 8192      | 16384 | 32768     | 65536    |
| ----- | ----------- | ----- | ---- | ---- | --------- | ----- | --------- | -------- |
| x1    | SimdFFT     | 2.2G  | 2.7G | 3.3G | 4.4G      | 4.8G  | 5.3G      | 5.0G     |
| x1    | SyclFFT:CPU | 3.0G  | 4.6G | 4.7G | 4.9G      | 4.9G  | 5.0G      | 4.9G     |
| x1    | **WebGPU**  | 21.3M | 209M | 425M | 740M      | 1.4G  | 2.8G      | **6.1G** |
| x16   | SimdFFT     | 4.5G  | 4.8G | 5.3G | 4.6G      | 5.0G  | 5.0G      | 5.4G     |
| x16   | **WebGPU**  | 1.5G  | 3.3G | 4.0G | **11.0G** | 6.2G  | 6.4G      | 9.2G     |
| x128  | SimdFFT     | 4.8G  | 4.8G | 5.0G | 5.2G      | 5.0G  | 5.1G      | 1.9G     |
| x128  | **WebGPU**  | 2.3G  | 5.0G | 3.5G | 5.6G      | 10.3G | **13.1G** | 5.0G     |

Same shape as the native GPU story, but a much steeper entry cost: WebGPU is **~140× slower than in-browser CPU at
x1/N=1024** (21.3M vs 3.0G) and only breaks even around N=32768 unbatched. Batching pulls break-even right down —
at x16 it wins from N=8192, at x128 it peaks at 13.1G. Note the in-browser CPU rows sit at 2-5G against 11-17G
native, i.e. WASM costs the CPU path ~3-4×, so WebGPU's _relative_ win in the browser flatters it versus a native
SYCL comparison (native SYCL:GPU reached 76G at x128).

**Three browser-runner bugs fixed to get a trustworthy number** (all in `82db40cb`):

1. **Chrome was hardcoded `--headless=new`**, which reaches only SwiftShader — the real reason a hardware adapter had
   been thought unreachable, not an environment limit. `--headed` obtains a real adapter.
2. CDP treated a page unresponsive for 15 s as dead; a benchmark holds the main thread far longer between Asyncify
   yields (both `call()` and the _separate_ `evaluate()` default had to be raised — the latter shadowed the former).
3. Captured page output was discarded on PASS, throwing away the whole benchmark table.

**-O3 on the WASM build is load-bearing, not tidiness:** at the default -O0 the host rows ran ~100× slow while the GPU
rows were unaffected, and it was also what exhausted the 2 GiB WASM heap mid-sweep. Both symptoms vanished with -O3.

## 22. VEHICLE DECIDED (maintainer, 2026-08-19) — split by backend, and the code checks out

**The answer, verbatim:** _"Keep the regular per-block and (managed) sub-Graph dispatch for SYCL, for WebGPU/GLSL rely
on the managed sub-Graph pattern where there in-built scheduler also takes care of the dispatch and uses merely the Edge
information to see the connection between blocks but does not necessarily need to rely on the CircularBuffer<T>."_

Plus: **Q2 = skip fill + debug-fill flag**, **Q3 = both backends together**. Q1 was not asked — §21.1 made it moot.

This is a FOURTH vehicle, not V3/V1/§21.5 as written. It supersedes §20.2's ranking for the shader backends:

- **SYCL keeps both models** — per-block dispatch AND the managed sub-Graph. §21.5's conflict (iii) ("SYCL would then
  have two execution models — decide or unify") is answered: **keep both, do not unify.** SYCL keys on
  `isDeviceAccessible(span.data())` and never consults a region.
- **GLSL/WebGPU get the region only.** The inner scheduler dispatches, reads `Edge` for topology, and interior edges
  need **no `CircularBuffer<T>` at all** — it owns a device buffer per interior edge instead.
- **The ring-identity mirror (V3) is therefore NOT built.** With no host ring behind an interior edge there is no
  ring to key on, no high-water mark over ring indices, and **§20.7's top risk — ring wrap — disappears by
  construction.** That was the design's biggest hazard and this choice deletes it rather than mitigating it.
- **§20.4 Phase 0 is no longer gating.** The region scheduler drives its members itself and never returns to the
  parent between them, so one-encoder/N-dispatches/one-submit (shape iii) is available BY CONSTRUCTION, not as a
  conditional Phase 4. Measure (i) vs (iii) inside the region's own benchmark; do not build a separate probe.

### 22.1 STRUCTURALLY VERIFIED (2026-08-19) — the parent CAN drive a `ScheduledBlockGroup` synchronously, but only if the region writes its own `work()`

§21.5's conflict (ii) was _"`SchedulerWrapper` runs its inner scheduler on its own `std::thread`, which fights
single-threaded GPU submission and Asyncify"_. **RESOLVED, favourably — the thread is not part of the contract.**

- `SchedulerModel` is six virtuals (`SchedulerModel.hpp:23-38`): `setGraph`, `asBlockModel`, `start`, `stop`,
  `requestWorkQuiescence`, `releaseWorkQuiescence`. **No thread anywhere in it.** The `std::thread` lives only in
  `SchedulerWrapper::start()` (`SchedulerModel.hpp:78`), which the device region simply does not use.
- `graph::flatten()` defaults to `traverseCategory = TransparentBlockGroup` (`Graph.hpp:959`), so it does **not
  descend** into a `ScheduledBlockGroup` — but the group block ITSELF is added (`:964`, filter defaults to `All`).
  It therefore lands in the job list (`Scheduler.hpp:1543-1547`) and `traverseBlockListOnce` calls **`work()` on it
  every iteration** (`Scheduler.hpp:624`).
- So the parent does BOTH: `asSchedulerModel(*block)->start()` at start-up (`Scheduler.hpp:673-685`, mirrored in
  `stop()` `:881-893`) **and** `work()` in the loop.
- **THE CATCH, and it is load-bearing: `Block<Derived>::work()` is INERT for any non-`NormalBlock` category** —
  `if constexpr (Derived::blockCategory != block::Category::NormalBlock) return {requestedWork, 0UZ, OK};`
  (`Block.hpp:2305-2311`). So the parent really does call `work()` on today's `SchedulerWrapper<Simple>` every
  iteration and it does nothing; the inner thread does the real work. That is why §21.6 measured a correct
  `dataSets=1024` with no race. **Synchronous drive is therefore NEW behaviour to implement, not a property to lean on.**
- **The implementation is local and needs no core change.** `BlockWrapper::work()` (`BlockModel.hpp:772`) forwards as
  `blockRef().work(requested_work)` — a NON-VIRTUAL call on the concrete type — so a region type that declares its own
  `work(std::size_t)` **shadows** `Block<Derived>::work()` and is the one called. The region defines `work()`; the
  category check never runs. (Name-lookup claim: high confidence, but the probe must prove it dynamically, not by
  reading.) Do NOT relax the category check in `Block.hpp` — that would wake every existing group.
- Two obligations that follow: (1) `asSchedulerModel()` must return non-null or the parent emits _"ScheduledBlockGroup
  is not a SchedulerModel"_ at start AND stop; (2) our `start()` must set the group's own lifecycle to RUNNING —
  the parent deliberately skips `changeStateTo(RUNNING)` for this category (`Scheduler.hpp:684`).

### 22.2 `Edge` already carries everything; `EdgeState::Overridden` is the ready-made mark

`Edge` (`BlockModel.hpp:74-96`) holds source/destination `BlockModel` + `PortDefinition`, `_domain`, `_minBufferSize`
and `_state` — full topology with **no buffer implied**. And `EdgeState::Overridden` (`BlockModel.hpp:75`) exists, is
used nowhere, and already makes `connectPendingEdges` skip an edge (`Graph.hpp:854`). That is precisely the mark for
_"this interior edge has no host ring; the region owns it"_ — no new enum, no signature change.

### 22.3 Consequences to settle before code

- **Region formation** (§21.5 conflict (i), still open): hand-built managed sub-Graph, as today, or auto-formed from
  maximal same-domain connected subgraphs at connect (the predicate exists, `Graph.hpp:734-736`). Proposal:
  hand-built first — it is the proven vehicle and `b0fcf8e0` just made it work; auto-formation is a follow-up that
  restores R1 without changing the runtime.
- **Tags do not cross the interior.** Interior edges have no tag ring; kernel-facing spans already lack tag members
  (`ef6f2a16`, `27b1f48c`). Tags ride the boundary only — consistent with §18.1 decision 2. State it, test it.
- **Skip-fill + debug flag** means: interior edge contents are staged back to host only when the flag is set, so a
  tap/monitor/bisect run can still see real data. Default off.
- **GL needs the barrier, WebGPU does not** (§20.3): `glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)` between two
  dispatches on the same SSBO is mandatory; WebGPU gets ordering and visibility free from `GPUQueue.submit`.
- **Requirement (b) (WAR/fan-out) is still real** and the PoC does not de-risk it (§20.8). In a region the rule is
  the same: a buffer with more than one consumer is never written in place. The region owns the edge buffers, so it
  can enforce this at region-build time from `_edges` — cheaper here than in V3.

### 22.4 STEP DONE — the vehicle is PROVEN (2026-08-20, `core/test/qa_DispatchGroup.cpp`, gcc15 + clang20 green)

§21.5's cheapest probe, ~80 lines, **no device code**: a minimal standalone `PassthroughDispatchScheduler` that
reports `ScheduledBlockGroup`, implements the six `SchedulerModel` virtuals with a thread-free `start()`, and does its
member work in **its own `work()`** (per 22.1 — the shadowing is the thing under test). Wire it exactly as the D1 probe
(`source → group[Copy] → sink`). **RED before `b0fcf8e0`, GREEN after** — it settles the whole vehicle before any
GLSL/WGSL is written.

**Assert more than `sink.count > 0`** — that is the shape of assertion D1 punished (it passed on structure while data
never flowed). Three assertions, all deterministic, no sleeps:

1. a counter incremented inside the group's own `work()` is **non-zero** — proves the shadowing actually took;
2. `sink.count` advances **only in step with** that counter — proves the data moved _through_ the region, not around it;
3. `std::this_thread::get_id()` captured inside the group's `work()` **equals** the parent worker's — proves
   synchronous drive on the parent's thread rather than inferring it from the absence of a thread.

### 22.5 Quiescence — a synchronous region satisfies it almost trivially

`requestWorkQuiescence()` (`Scheduler.hpp:259-265`) sets a flag then **spins until `_nWorkersInWork == 0`**; the guard
(`:291-296`) fans it out to every nested `SchedulerModel`, and it wraps graph mutation and resume (`:920`, `:1142`,
`:1193`). When the region runs on the parent's own thread its `work()` is by construction not executing at the moment
the parent requests quiescence, so the wait is empty. Still implement it honestly — flag + own in-work counter, ~6
lines — because a message-driven mutation or `resume()` can arrive from another thread under `multiThreaded`.

### 22.6 Asyncify does NOT dissolve — it moves

§20.7's UNVERIFIED item was _"whether Asyncify tolerates V3's flush points mid-`work()`"_. In the region model the
submit+wait sits mid-**parent**-`work()` instead. Same open question, new location. Keep it flagged; the browser leg
of the region benchmark is where it gets answered.

### 22.7 PROBE RESULT (2026-08-20) — vehicle CONFIRMED, and it corrected two of my own claims

`core/test/qa_DispatchGroup.cpp` (new, ~190 lines incl. both tests; registered in `core/test/CMakeLists.txt`).
`PassthroughDispatchScheduler` = a `ScheduledBlockGroup` holding two `Copy<float>` members joined by an interior
edge, boundary ports exported as `inExp`/`outExp`, driven by `Simple<externalStep>` — **no threads, no sleeps, no
deadline**, the test thread itself calls `step()`. `DispatchWrapper` = `SchedulerWrapper` minus the `std::thread`.

**Numbers, 64 `step()` calls:** `dispatchCalls=64` · `memberCalls=128` · `sinkCount=4,128,768`. Exactly one dispatch
per step and two member calls per dispatch — the parent's loop is what drives the region, 1:1.

**The negative control is what makes this evidence.** Rename the region's own `work()` so the base is found instead:
`dispatchCalls=0` · `memberCalls=0` · **`sinkCount=0`** — the whole graph moves nothing, silently, with no error
anywhere. That is G21 reproduced on demand, and it is the RED half of the pair.

Proven, in order of what was actually at risk:

1. **The shadowing works.** A group declaring its own `work(std::size_t)` gets called through
   `BlockWrapper::work()`'s non-virtual forward; `Block<Derived>::work()`'s category short-circuit never runs.
2. **Data really flows through it** — quiescence test: silence the region, let the backlog drain, and the sink stops
   dead and stays dead; release quiescence and it resumes. The path goes THROUGH our `work()`, not around it.
3. **Synchronous on the caller's thread, FOR `externalStep`** — `_dispatchThread == std::this_thread::get_id()` of
   the test, asserted rather than inferred from the absence of a thread. **Scope it honestly:** this proves the region
   spawns no thread of its own and runs inline on whoever calls `step()`. It does NOT cover `multiThreaded` or
   `singleThreaded`, where the caller is a pool worker and `requestWorkQuiescence` may arrive from another thread —
   exactly the case §22.5 says needs a real flag+counter (the probe's `_quiescent` is a plain `bool`, sound under
   `externalStep` only). Asyncify remains untested; §22.6 stands unchanged.
4. **The parent's boundary bindings survive by construction** — no `SchedulerBase`, so no start-time port sweep.

**Two corrections to §22 as first written:**

- **§22.4's "RED before `b0fcf8e0`, GREEN after" was wrong.** This probe is INDEPENDENT of the sweep fix: not
  inheriting `SchedulerBase` means `disconnectAllEdges()` never runs, which is precisely why the vehicle works.
  `b0fcf8e0` remains necessary for `SchedulerWrapper`-based sub-graphs (§21.6) — just not for this one.
- **A group cannot use static ports.** `BlockWrapper::dynamicPortsLoader()` registers static ports **only when
  `TBlock::blockCategory == NormalBlock`** (`BlockModel.hpp:690`). A `ScheduledBlockGroup` declaring `PortIn`/`PortOut`
  gets no dynamic ports at all, so `graph.connect` cannot find them and no ring is ever made. Boundary ports MUST come
  from `GraphWrapper::exportPort` — i.e. the framework only supports the managed sub-Graph shape here, which is
  exactly what the maintainer chose. Recorded as the second half of G21.

**Also required of a group (found by compiler, not by reading):** `GraphWrapper` forwards `blocks()`, `edges()` and
`findPortInBlock` to the wrapped type, so the dispatch type must expose `graph()`, `blocks()` and `edges()`;
and `gr::Graph`'s move-assignment is deleted (`Graph.hpp:382`), so `setGraph` must rebuild the `meta::indirect`.

**Gated:** gcc15 `build-gcc15-debug` and clang20 `build-ci-clang20-debug`, both **24 asserts / 2 tests, rc=0**, no
warnings under the project `-Werror` set. The 24/2 count also discharges G10's silent-suite-drop hazard (a dropped
suite reads `0 asserts in 0 tests`, exit 0); the explicit `int main()` follows `qa_Scheduler`/`qa_Embedded` — this
repo's `ut` target supplies no `main` of its own, and omitting it fails to link. `clang-format-18` applied. **No production file was touched** — the change
is one new test plus one line in `core/test/CMakeLists.txt`, so no regression surface exists.

### 22.8 NEXT (not started)

With the vehicle settled, the open items are, in order: (a) region formation — hand-built vs auto-formed from
same-domain connected subgraphs; (b) interior edges without host rings — mark them `EdgeState::Overridden` so
`connectPendingEdges` skips them (`Graph.hpp:854`) and have the region own a device buffer per edge; (c) the
skip-fill debug flag; (d) the GL `glMemoryBarrier` / WebGPU no-barrier split; (e) requirement (b) fan-out, enforced
at region-build time from `_edges`.

## 23. REGION FORMATION — decided 2026-08-20

**Maintainer decisions:**

- **(1) a helper that wraps named blocks** — `makeRegion`-style: the caller names the members, the helper does the
  plumbing the probe does by hand (inner graph, wrapper, export the boundary ports, hand back their names). The DSP
  code is untouched, so R1 holds for the block author, and no core `Graph` surgery is needed.
- **(2) auto-formation is a NOTED FUTURE FEATURE, not now** — _"might be a worthwhile feature once the other primary
  path has been established"_. It is what fully restores R1 (write a flat graph, get device execution for free); the
  predicate already exists (`Graph.hpp:734-736`). Revisit once the helper path is proven end to end. Do NOT start it.
- **Vertical stack uses TODAY's host-span hatches first** (`processBulk_glsl`/`_webgpu`, `fft.hpp:224-279`): correctness
  gate before elision. It will show no throughput win — each member still round-trips the host at ~38 ns/sample — and
  the device-buffer hatch that removes that is the measured follow-up, not part of this step.
- **GPU figures: the maintainer pins the clock** (`sudo nvidia-smi -lgc 1500,1500`, release `-rgc`) and says when the
  card is idle; measurement waits for that rather than publishing unpinned numbers.

**Scope note recorded so it is not re-litigated:** "reproduce the FFT figures including WebGPU/GLSL" = **GLSL natively**
on the RTX 3070 via EGL + **WebGPU in headed Chrome**. There is no GLSL-in-browser leg to reproduce — GL 4.3 compute
cannot reach a browser (WebGL2 is GLES 3.0: no compute shaders, no SSBOs).

**Two facts that shape the vertical stack:**

- A **one-member region has no interior edge**, so it can only ever show plumbing overhead, never elision. Any
  demonstration of the region's value needs a **>=2-block device chain**.
- `bm_FFT_backends` measures the **algorithm layer**, not the block/graph path, so the region work cannot move those
  numbers. The block/graph path is what the vertical-stack test covers, and it needs its own baseline.

### 23.1 REGION FORMATION — LANDED (2026-08-20), gcc15 + clang20 green

`core/include/gnuradio-4.0/device/DeviceRegion.hpp` (new, backend-free — includes only `Block/Graph/SchedulerModel`,
so an R2 CPU-only user pays nothing) + `core/test/qa_DeviceRegion.cpp` (new, **35 asserts / 3 tests**).
`qa_DispatchGroup` is KEPT as the framework-contract characterisation test (24/2) — deliberately independent of
`DeviceRegion.hpp`, so it still guards G21 if our own header changes.

**Names (flagged for approval, easy to change):** `gr::device::Region` (the block group), `gr::device::RegionWrapper<T>`
(the thread-free `SchedulerWrapper`), `gr::device::makeRegion(gr::Graph&&) -> std::expected<RegionHandle, Error>`.

**What the helper does:** the caller emplaces members and connects the interior edges as usual and hands the graph
over; `makeRegion` exports **every port no interior edge claims**, sets `disconnect_on_done=false` on each member
(their boundary peers are attached by the parent), and returns the block plus its boundary port names. One boundary
port per direction is named `in`/`out`, several are `in0`,`in1`,… — so from the parent's side the common case reads
exactly like an ordinary block. Port _collections_ are refused with a clear error rather than silently mis-exported.

**Members run in topological order, not emplacement order.** `Region::topologicalOrder()` (Kahn; a feedback-loop
remainder keeps emplacement order, costing latency not correctness) — there is no topological sort in `Graph.hpp` to
reuse, only `detectFeedbackLoops`. **Verified by control:** with the sort replaced by `blocks()` order, a 3-member
chain emplaced forwards reaches the sink in 2 steps and the same chain emplaced backwards takes 4 — the test FAILS.
With the sort, both take 2. Without that control the assertion would have passed for the wrong reason.

**Still open here:** interior edges are ordinary host rings today. Marking them `EdgeState::Overridden` so
`connectPendingEdges` skips them (`Graph.hpp:854`) and letting the region own a device buffer per edge is §22.8(b),
untouched — and it is what makes a region worth more than plumbing.

### 23.2 NATIVE FFT BENCHMARK — HARNESS REPRODUCES, GPU FIGURES DO NOT (2026-08-20, contended card)

`build-acpp` (Release, GL ON), one 12 s run of `bm_FFT_backends`: **105 PASS / 0 FAIL**, matching §21.7 exactly, all
five backends detected (SimdFFT · SyclFFT:CPU · SYCL:GPU · SYCL:CPU · GLSL:GPU). Clock 1980 of 2100 MHz — but the card
was at **95 % utilisation with 6.4 GB held by an unrelated long-running job**, by arrangement with the maintainer.

N=4096, ops/s, recorded idle table (§21.7) vs this contended run:

| batch | backend      | §21.7 (idle) | now (95 % contended) | ratio    |
| ----- | ------------ | ------------ | -------------------- | -------- |
| x1    | SimdFFT      | 16.5G        | 13.7G                | 0.83     |
| x1    | SyclFFT:CPU  | 11.8G        | 9.0G                 | 0.76     |
| x1    | SYCL:CPU     | 1.3G         | 953M                 | 0.73     |
| x1    | **SYCL:GPU** | 4.1G         | **101M**             | **0.02** |
| x1    | **GLSL:GPU** | 1.6G         | **99.4M**            | **0.06** |
| x16   | SimdFFT      | 12.7G        | 15.7G                | 1.24     |
| x16   | SYCL:CPU     | 6.7G         | 5.4G                 | 0.81     |
| x16   | **SYCL:GPU** | 41.0G        | **16.3G**            | **0.40** |
| x16   | **GLSL:GPU** | 15.7G        | **1.4G**             | **0.09** |
| x128  | SimdFFT      | 15.3G        | 15.8G                | 1.03     |
| x128  | SYCL:CPU     | 10.5G        | 4.5G                 | 0.43     |
| x128  | **SYCL:GPU** | 76.3G        | **12.9G**            | **0.17** |
| x128  | **GLSL:GPU** | 14.0G        | **2.7G**             | **0.19** |

**The internal control does its job.** The CPU rows land within ~25 % of the record (some above, some below — noise,
not drift), so the machine and the harness are the same. The GPU rows are down **2.5x to 40x**. A contended card is
the only variable that moved, so the split is attributable, and the honest conclusion is:

- **the harness reproduces** — same backends, same pass count, same shape (GPU only wins once the batch amortises
  launch and transfer; at x1 SimdFFT still beats everything);
- **the GPU numbers are NOT a reproduction of §21.7** and must not be quoted as one. They are a contended floor.
  Re-run on an idle card to close this out; it costs 12 s.

**Not yet run: the browser/WebGPU leg** (`tools/webgpu/run_fft_benchmark.sh`). It needs headed Chrome on the live
display and the same GPU, so under contention it would produce equally uncomparable numbers while taking over the
maintainer's screen. Deferred to the idle-card window, where it is one more short burst.

### 23.3 VERTICAL STACK — FFT IN A REGION, HOST PATH GREEN (2026-08-20)

`blocks/fourier/test/qa_FFTRegion.cpp` (new, **21 asserts / 2 tests**, clang20 `build-ci-clang20-debug`).
`CountingSource<complex<float>>` -> `region[Copy<complex<float>> -> FFT<float>]` -> `TagSink<complex<float>>` under
`Simple<externalStep>`, compared sample-by-sample against the identical FFT in a **flat** graph.
Result: **bit-identical**, max|difference| = 0 over >= 1024 complex samples. A real DSP block, not a `Copy`, keeps its
contract inside a region.

Two things worth recording:

- `FFT<float>` is the **stream-mode** specialisation (`fft.hpp:55`), `PortIn/PortOut<std::complex<float>>` — `T` is the
  PRECISION, not the input domain. It is the only specialisation carrying all three hatches (`processBulk_sycl`
  `:140`, `_glsl` `:177`, `_webgpu` `:224`), each `requires std::same_as<T, float>`. The `FFT<T, DataSet<P>>`
  specialisation has the SYCL hatch only — so the shader vertical stack must use stream mode.
- The equality assertion needed a **non-vacuity guard**: two all-zero spectra also compare equal. `maxMagnitude(flat)
  > 0` now rules that out. Same class of mistake as D1's structure-only assertion.
- `build-gcc15-debug` has the block registry OFF, so `TARGET gnuradio4::GrFourierBlocksShared` is false and
  `blocks/fourier/test` is never added there. **`build-ci-gcc15-release` does build it** once regenerated — the dir
  was merely stale, so the suite IS gated on both compilers like every other in this batch.

### 23.4 TERMINATION — NO GAP (2026-08-20, measured not assumed)

Concern raised in review: `Region::work()` reports `DONE` only when every member does, but `makeRegion` sets
`disconnect_on_done=false` on all members — so could a region make a graph un-endable? **No.** `qa_FFTRegion`'s
third test drives `CountingSource{n_samples_max=4*fft_size} -> region[Copy->FFT] -> sink` and asserts the parent's
`step()` reaches `work::Status::DONE` within 512 steps. It does, on gcc15 and clang20. A bounded source still ends a
run that goes through a region; a region is not restricted to a step budget.

### 23.5 SHADER VERTICAL STACK — WRITTEN, SKIPPING, NEEDS A GL-ON + REGISTRY-ON BUILD

The GLSL leg exists (`FFTRegionDevice` suite): it registers the runtime, checks `DeviceContextRegistry::tryResolve`
actually serves `gpu:glsl`, and compares the in-region device spectrum with the flat host spectrum to a 1e-3 relative
bound (a shader FFT is a different implementation, so exact equality is the wrong assertion there). It **skips
honestly** rather than falling back to the CPU and reporting a device pass.

**It skips everywhere locally, and the build matrix says why:**

| dir                      | GL     | registry | fourier tests | can run the GLSL leg                                            |
| ------------------------ | ------ | -------- | ------------- | --------------------------------------------------------------- |
| `build-ci-gcc15-release` | OFF    | ON       | yes           | no — `registerGlslRuntime()` is the stub (`GlslRuntime.hpp:37`) |
| `build-ci-clang20-debug` | OFF    | ON       | yes           | no — same                                                       |
| `build-acpp`             | **ON** | OFF      | **no**        | no — `blocks/fourier/test` is never added                       |

**RESOLVED 2026-08-21 — see §26.**

## 24. RENAMED Region -> Domain (maintainer, 2026-08-20), and the FFT numbers

**Rename done**, all suites re-gated green on gcc15 + clang20: `gr::device::Domain` · `DomainWrapper<T>` ·
`makeDomain(gr::Graph&&) -> std::expected<DomainHandle, Error>` · `device/DeviceDomain.hpp` · `qa_DeviceDomain`
(35/3) · `qa_FFTDomain` (29/3 + a self-skipping device suite) · `qa_DispatchGroup` (24/2, comments updated).
Rationale: "Domain" is already the term in `ComputeDomain` and related contexts. The header states the distinction
so the two never blur: **`gr::ComputeDomain` is the VALUE naming a backend ("gpu:glsl"); `gr::device::Domain` is the
GROUP of blocks sharing one.**

### 24.1 FFT PERFORMANCE — measured 2026-08-20, same machine, same session

**Plain algorithm** (`bm_FFT_backends`, CPU rows uncontended; ops/s at the 5·N·log2(N) convention, converted to
samples/s here for comparability):

| N    | batch | SimdFFT ops/s | => MS/s   |
| ---- | ----- | ------------- | --------- |
| 1024 | x1    | 14.1G         | ~282 MS/s |
| 4096 | x1    | 13.7G         | ~228 MS/s |

**Integrated in a Block/Graph** (`bm_fft_subgraph`, gcc15 Release, `FFT<float, DataSet<float>>` fft_size=1024,
1024 DataSets = 1 Mi samples, three independent runs):

| run | flat graph | inside a Domain | delta      |
| --- | ---------- | --------------- | ---------- |
| 1   | 42.74 MS/s | 39.78 MS/s      | +7.5 %     |
| 2   | 38.80 MS/s | 41.15 MS/s      | **-5.7 %** |
| 3   | 40.02 MS/s | 40.82 MS/s      | -1.9 %     |

**Two readings, both worth keeping:**

1. **The Domain vehicle costs nothing measurable.** The sign of the "overhead" FLIPS run to run — the sub-graph is
   _faster_ than flat in run 2. The effect is below the noise floor at n=3, which is consistent with §21.8's 2-4 %
   and further weakens it. Quote **"no measurable overhead for one member"**, not a percentage.
2. **In-graph is ~40 MS/s against ~282 MS/s for the raw algorithm — but that ~7x is NOT all framework.**
   `bm_fft_subgraph` runs **spectrum mode**: real->complex transform PLUS window, magnitude, phase (`atan2`) and a
   `DataSet` allocation per transform. `bm_FFT_backends`' SimdFFT row is a bare complex->complex transform. The gap
   is framework overhead AND spectrum-mode work, conflated. **Do not attribute it to the framework.**
   Decomposing it needs one stream-mode (`FFT<float>`, complex->complex) in-graph run — same work as the algorithm
   row, so the difference would then be framework alone. NOT YET MEASURED; that is the honest next step.

### 24.2 WebGPU — what is integrated and what is not

**Integrated and benchmarked** (`3a1f96a9`, `82db40cb`): a third backend beside SYCL and GLSL through the same
domain resolution and dispatch, WGSL Stockham FFT (`WgslFFT.hpp`), the `processBulk_webgpu` hatch
(`fft.hpp:224`, `requires std::same_as<T, float>`), registry **withdrawal** on browser device loss, and a WebGPU row
in `bm_FFT_backends` via `tools/webgpu/run_fft_benchmark.sh`. Hardware-verified: `vendor=nvidia architecture=ampere`.
Recorded browser figures (idle card, WASM -O3, §21.10): x1/N=1024 **21.3M** ops/s — ~140x slower than in-browser CPU
— breaking even near N=32768 unbatched; at x16 it wins from N=8192; x128 peaks at **13.1G** (N=32768).

**NOT integrated: the WebGPU leg of the Domain vertical stack.** The GLSL leg is written and self-skipping; the
WebGPU one is not written. It needs an Emscripten build of the suite plus headed Chrome, so it belongs with the
idle-card window rather than the current session. No design work is outstanding for it — the hatch and the domain
plumbing both exist.

## 25. ALL-BACKEND PLAIN-ALGORITHM SWEEP (2026-08-20) — one table, seven backends, contended card

Native `bm_FFT_backends` in `build-acpp` (best of 3) + browser `tools/webgpu/run_fft_benchmark.sh` headed (best of 3)

- one headless run for the SwiftShader comparison. 105 PASS native, 64 PASS browser, every run rc=0.

**batch x1** — ops/s at the 5·N·log₂(N) convention

| backend                       | 1024  | 2048  | 4096  | 8192  | 16384 | 32768 | 65536 |
| ----------------------------- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| SimdFFT                       | 11.6G | 13.3G | 16.9G | 17.1G | 17.0G | 16.2G | 16.0G |
| SyclFFT:CPU                   | 11.4G | 11.3G | 12.1G | 11.8G | 12.3G | 10.9G | 11.0G |
| SYCL:CPU                      | 367M  | 702M  | 1.0G  | 1.5G  | 2.2G  | 3.3G  | 5.5G  |
| SYCL:GPU                      | 16M   | 40M   | 4.9G  | 1.3G  | 4.7G  | 5.4G  | 19.4G |
| GLSL:GPU                      | 21M   | 138M  | 612M  | 545M  | 1.1G  | 1.0G  | 1.7G  |
| SimdFFT (browser)             | 2.1G  | 2.7G  | 3.3G  | 3.9G  | 4.7G  | 5.3G  | 4.4G  |
| WebGPU:GPU (browser)          | 13M   | 41M   | 89M   | 221M  | 430M  | 901M  | 1.8G  |
| WebGPU:SwiftShader (headless) | 12M   | 43M   | 96M   | 203M  | 407M  | 568M  | 613M  |

**batch x16** — ops/s at the 5·N·log₂(N) convention

| backend                       | 1024  | 2048  | 4096  | 8192  | 16384 | 32768 | 65536 |
| ----------------------------- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| SimdFFT                       | 12.0G | 12.4G | 12.9G | 12.6G | 15.7G | 15.3G | 15.7G |
| SyclFFT:CPU                   | 11.6G | 11.2G | 11.6G | 11.5G | 11.9G | 11.8G | 12.0G |
| SYCL:CPU                      | 3.8G  | 7.0G  | 8.5G  | 9.3G  | 12.5G | 8.8G  | 13.5G |
| SYCL:GPU                      | 12.6G | 23.5G | 36.4G | 18.3G | 55.6G | 41.1G | 31.4G |
| GLSL:GPU                      | 1.0G  | 8.2G  | 17.6G | 5.4G  | 7.1G  | 5.1G  | 4.6G  |
| SimdFFT (browser)             | 4.8G  | 4.9G  | 5.5G  | 5.2G  | 4.7G  | 5.3G  | 5.6G  |
| WebGPU:GPU (browser)          | 389M  | 885M  | 1.4G  | 3.1G  | 5.8G  | 6.1G  | 7.3G  |
| WebGPU:SwiftShader (headless) | 331M  | 497M  | 613M  | 705M  | 762M  | 739M  | 764M  |

**batch x128** — ops/s at the 5·N·log₂(N) convention

| backend                       | 1024  | 2048  | 4096  | 8192  | 16384 | 32768 | 65536 |
| ----------------------------- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| SimdFFT                       | 15.5G | 15.4G | 15.7G | 15.8G | 15.8G | 15.2G | 15.1G |
| SyclFFT:CPU                   | 11.3G | 11.2G | 11.7G | 11.5G | 11.8G | 10.9G | 11.1G |
| SYCL:CPU                      | 11.6G | 12.6G | 12.5G | 8.3G  | 14.0G | 8.7G  | 9.0G  |
| SYCL:GPU                      | 27.8G | 7.8G  | 68.6G | 53.5G | 52.3G | 39.7G | 58.2G |
| GLSL:GPU                      | 726M  | 1.7G  | 5.9G  | 4.3G  | 3.7G  | 4.3G  | 4.2G  |
| SimdFFT (browser)             | 5.3G  | 5.0G  | 5.2G  | 5.4G  | 5.2G  | 5.3G  | 1.9G  |
| WebGPU:GPU (browser)          | 2.4G  | 4.9G  | 5.0G  | 6.0G  | 9.5G  | 9.7G  | 4.4G  |
| WebGPU:SwiftShader (headless) | 685M  | 763M  | 781M  | 762M  | 754M  | 745M  | 10.1G |

**HEADLESS IS SWIFTSHADER — measured this session, both modes minutes apart on this box:** headless reports
`WebGPU: (swiftshader)`, headed reports `WebGPU: (ampere)`. Chrome's hardware WebGPU support is not in question
(the adapter names the RTX card); it is _headless mode here_ that does not reach it. `run_fft_benchmark.sh` already
encodes this — it passes `--headed --require-hardware` by default and refuses a software adapter unless
`--allow-software` is given. **There is no Firefox path**: `run_wasm_webgpu.py` is CDP/Chrome only (zero Firefox
references), so a Firefox leg means a WebDriver BiDi driver plus Firefox's WebGPU flags — real work, not a switch.

**Trust boundary for this table:**

- **CPU rows are sound** — they never touch the GPU. Native SimdFFT 15-17G, in-browser SimdFFT ~5G, i.e. **WASM
  costs the CPU path ~3x**, reproducing §21.10's finding.
- **Every GPU row is contended** (the card sat at 98 % with 6.4 GB held by an unrelated job throughout). Best-of-3
  recovers a lot for SYCL:GPU (x128/N=4096 **68.6G** vs 76.3G recorded idle = 90 %), far less for the others:
  WebGPU x128/N=32768 reads **9.7G** vs 13.1G recorded, x1/N=1024 **13M** vs 21.3M. GLSL is the most erratic —
  x16/N=4096 17.6G _exceeds_ the 15.7G record while x128/N=4096 5.9G sits far below 14.0G.
- **Two visible artefacts, do not read them as signal:** SYCL:GPU x128/N=2048 = 7.8G between 27.8G and 68.6G
  neighbours, and SwiftShader x128/N=65536 = 10.1G, which is impossible for a software rasteriser. Both are
  contention/scheduling artefacts of a single sample surviving a best-of-3.

**One real effect the table does show, independent of contention:** SwiftShader **beats** hardware WebGPU at x1 for
N<=4096 (96M vs 89M at N=4096). That is not noise — it is the per-dispatch browser round trip. Software runs
in-process and pays no submit/map round trip, so at tiny work quanta the round trip, not the arithmetic, is the
whole cost. It is the same ~2.2 ms floor the WebGPU block path documents, seen from the other side.

**Conclusion unchanged:** the shape reproduces (GPU only wins once batch amortises launch and transfer; at x1 the
native CPU SimdFFT beats every GPU backend at every size), but **these GPU magnitudes are a contended floor, not a
reproduction of §21.7/§21.10.** One idle-card window closes it: 3x12 s native + 3x12 s browser.

## 26. GLSL VERTICAL STACK — RUNS, AND IT IS PROVABLY NOT A CPU FALLBACK (2026-08-21)

`cmake -S . -B build-ci-gcc15-release -DGR_ENABLE_GL_COMPUTE=ON` (maintainer-approved), 101 s rebuild of
`qa_FFTDomain`. The `FFTDomainDevice` suite goes from **0 asserts (skipped)** to **16 asserts / 1 test, all passing**:
`CountingSource<complex<float>>` -> `Domain[Copy -> FFT<float> @ compute_domain="gpu:glsl"]` -> `TagSink`, compared
against the identical FFT in a flat host graph.

**The number that matters: relative difference = 9.7e-12, NON-ZERO** (peak 4.23e9 over 4,128,768 compared samples).
That single fact is what makes this evidence rather than a green tick:

- the host test already asserts a Domain adds **exactly zero** difference on the CPU path, so
- a silent CPU fallback here would produce a **bit-identical** result, i.e. 0.0, and
- 9.7e-12 != 0 therefore proves a **different implementation** computed the spectrum — the GLSL Stockham kernel on
  the RTX 3070 — while still agreeing with the host to well inside the 1e-3 bound.

**The check is now permanent, not a one-off**: the suite asserts `relative > 0.0` with the rationale in place —
_"a bit-identical result means a silent CPU fallback, not a GLSL run"_. Without it this leg could pass for ever while
testing nothing, which is the `qa_DeviceSeam`/D1 failure mode a third time.

**Portability held:** in the GL-OFF dirs (`build-ci-clang20-debug`) the leg still compiles and **skips honestly**
(`registerGlslRuntime()` is the stub, `GlslRuntime.hpp:37`) rather than failing or silently CPU-passing. So the suite
is safe in a CI lane without GL.

**Status of the three legs of the vertical stack:**

| leg                   | state                                                                                            |
| --------------------- | ------------------------------------------------------------------------------------------------ |
| host CPU              | GREEN — bit-identical to flat, gcc15 + clang20 (29/3)                                            |
| **GLSL (native GPU)** | **GREEN — 16/1, fallback-discriminated, `build-ci-gcc15-release` with GL ON**                    |
| WebGPU (browser)      | NOT WRITTEN — needs an Emscripten build of the suite + headed Chrome; no design work outstanding |

**Note for the matrix:** only `build-ci-gcc15-release` now has GL ON; `build-ci-clang20-debug` does not, so the GLSL
leg has a **single-compiler** gate today. Enabling GL there too would close that, and is one configure flag.

## 27. INTERIOR DEVICE BUFFERS + THE TAG DROP — root-caused 2026-08-21, and the plan changed

### 27.1 "Interior device buffers" is TWO tasks, one already shipping and one blocked

**SYCL: already implemented, and now PROVEN to work inside a Domain.** `Graph::applyEdgeConnection` already
(a) auto-populates an edge's domain from the members' `compute_domain` (`Graph.hpp:701`), (b) marks an edge whose two
endpoints share ONE device domain as `Access::DeviceOnly` (`:734`), and (c) resolves its data resource to device USM,
**authoritative over every other tier** (`:759-776`). That is PR-F's measured 1.45-1.47x win — nothing to build.
The open question was whether this still happens for a Domain, whose interior edges are connected by its OWN graph
rather than the parent's. **It does:** `qa_DeviceDomain` now asserts an interior edge between two `gpu:sycl` members
is `isDevice()` and `Access::DeviceOnly` after `startDispatch()`. Control: point one member at `host` and the
assertion FAILS, so it discriminates. 45 asserts / 4 tests, gcc15 + clang20.

**GLSL/WebGPU: blocked by construction.** An SSBO / `WGPUBuffer` is an opaque handle, not host-dereferenceable, so it
can never be a `std::pmr::memory_resource` backing a `CircularBuffer`. Verified: **no shader backend registers a PMR
provider at all** — only `SyclRuntime`/`UsmMemoryResource`/`CudaVmmMemoryResource` define one. So giving a shader
Domain its own interior buffers requires the **device-handle hatch** (a second hatch signature taking buffer handles
instead of host spans) — the very thing deferred in §23 in favour of the existing host-span hatch.
**Consequence: do NOT implement "interior device buffers" as a new mechanism.** For SYCL it exists; for shaders it is
gated on a deferred prerequisite. This is why the instruction misfired, and it is the single most useful thing to know.

### 27.2 The CPU-fallback tag drop — ROOT CAUSE

Not a fallback bug. The chain:

1. `qa_DeviceSeam`'s block forwards its custom tag **only inside `processBulk_sycl`** (`input.tags()` ->
   `output.publishTag()`); its CPU path is a bare `processOne` with no tag semantics at all.
2. The framework's automatic forwarder `forwardInputTags` (`Block.hpp:1174-1195`) is **filtered**: it propagates only
   keys starting with `gr:` or present in `settings().autoForwardParameters()`. The test's key is `"kind"`, which is
   neither — verified, nothing adds it to the supplement.
3. So when no device is served, `dispatchCpuFallback` runs the `processOne` loop, which copies samples and touches no
   tags, and the custom tag is gone.
   **Therefore: tag forwarding lives in the block's hatch, and the CPU path of the same block has none.** The framework
   cannot fix this generically for a `processOne`-only block, because `processOne` has no tag API to call.
   Note the current test does NOT reproduce it: with `ACPP_VISIBILITY_MASK=omp` a **`host:sycl`** device is still served,
   so the hatch still runs (verified: 12/1 green both with the GPU visible and hidden). The drop needs _no SYCL at all_.

### 27.3 The maintainer's requirement, answered in the three parts it has

_"tags or at least the block settings should be available also on/within device processing functions"_:

- **expert hatches (`processBulk_sycl`/`_glsl`/`_webgpu`): tags ARE already available** — `qa_DeviceSeam` reads
  `input.tags(count)` and calls `output.publishTag(...)` inside the hatch today. Nothing to do.
- **settings: ALREADY available on device as read-only scalars** (§18.1 decision 3, `DeviceBlockState` deleted). The
  "or at least" half of the requirement is met.
- **auto-parallel kernels: tags are NOT available, deliberately.** `Graph.hpp:752` keeps the tag axis host-accessible
  on purpose — _"tags are host-side metadata — a kernel may not publish them"_ — and `mergedInputTag()` returns a view
  aliasing host tag-ring memory that is UB to dereference in a kernel (F1, the same wound). This is a DESIGN DECISION
  to revisit, not an oversight; changing it needs a device-representable tag form (POD, no pmr string keys), which is
  a real design task, not a flag.

## 28. ACTIONABLE TODO — each item carries its blocker (2026-08-21)

> **SPENT — do not work from this list.** Every item was either completed (§30-§46) or removed with the shader
> backends. The live list is the OPEN table in the READ FIRST block, verified against the code 2026-08-25.

### A · actionable today, nothing in the way

| #      | item                                                                                                                                                                                                                                                                                                                          | size                                  |
| ------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------- |
| A1     | **SYCL leg of the Domain vertical stack** — residency marking is proven; now run a Domain-hosted SYCL chain on real hardware and confirm the USM resolution end to end. The one backend×Domain cell never exercised.                                                                                                          | small, needs acpp + a short GPU burst |
| ~~A2~~ | ~~GL memory barrier~~ **WRONG — ALREADY DONE (checked 2026-08-21).** Both dispatch sites already issue `glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)` immediately after `glDispatchCompute`: `GlComputeContext.hpp:254-255` and `DeviceContextGLSL.hpp:149-150`. The item came from §20.3's design note, not from the code. | nothing to do                         |
| A3     | **Two-compiler gate for the GLSL leg** — GL is ON only in `build-ci-gcc15-release`.                                                                                                                                                                                                                                           | one flag + rebuild                    |
| A4     | **Stream-mode in-graph FFT run** — decomposes framework cost from spectrum-mode cost, closing the "~7x but not all framework" caveat.                                                                                                                                                                                         | ~30 min                               |
| A5     | **Idle-card re-run of the all-backend sweep** — turns a contended floor into a reproduction.                                                                                                                                                                                                                                  | ~90 s, needs the card free            |
| A6     | **WebGPU leg of the Domain vertical stack** — emcc build of the suite + headed Chrome. No design work outstanding.                                                                                                                                                                                                            | medium                                |

### B · blocked on the deferred device-handle hatch — do NOT start these first

| #   | item                                                                         | blocker                                                             |
| --- | ---------------------------------------------------------------------------- | ------------------------------------------------------------------- |
| B1  | **Interior device buffers for GLSL/WebGPU**                                  | needs a hatch taking buffer handles, not host spans (§27.1)         |
| B2  | **Fan-out / WAR rule** (a buffer with >1 consumer is never written in place) | only meaningful once the Domain owns buffers, i.e. B1               |
| B3  | **skip-fill + debug-fill flag**                                              | only meaningful once interior edges stop having host rings, i.e. B1 |

### C · needs a maintainer decision, not code

| #   | question                                                                                                                                                                                                      |
| --- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| C1  | **CPU-fallback tag drop** — root-caused (§27.2). Options: accept it as "the hatch owns its tags"; or require a `processBulk` CPU twin whenever a hatch forwards tags; or make the divergence a compile error. |
| C2  | **Tags in auto-parallel kernels** — needs a device-representable tag form before it is even possible (§27.3).                                                                                                 |
| C3  | **PR re-cut** — mapping counts 56 against 63 commits + 2 new. Fresh backup branch first.                                                                                                                      |
| C4  | **`b0fcf8e0` message** still says "about +10%", superseded by "below the noise floor". Needs a rebase.                                                                                                        |

### D · pre-existing device gaps — RE-VERIFIED AGAINST THE CODE 2026-08-21 (most were already fixed)

**The §18.2 fix list and the OPEN section are substantially STALE — they date from 2026-07-09 and were never
re-checked. Five of seven items I had carried forward are already done.** Verify before working any of them.

| item                                                       | verified status                                                                                                                                                                                       |
| ---------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| A2 GL memory barrier between dispatches on one SSBO        | **DONE** — `GlComputeContext.hpp:254-255`, `DeviceContextGLSL.hpp:149-150` both issue it right after `glDispatchCompute`                                                                              |
| `DeviceSpans.hpp` @brief on the feature-test divergence    | **DONE** — already written, `DeviceSpans.hpp:18-20`                                                                                                                                                   |
| F10 per-`work()` mutex-guarded scheduler lookup            | **DONE** — `schedulerCache` resolves once per block and is reused, `ExecutionStrategy.hpp:170-188`                                                                                                    |
| F9 unbounded shader cache (work quantum baked into source) | **DONE** — `ShaderFragment.hpp:42`: "the element count is a uniform, not a baked literal: baking it would compile — and cache forever — a new program"                                                |
| F2 multi-GPU USM context mismatch                          | **DONE** — resources are keyed `(kind, deviceIndex)` with a per-queue USM resource, `SyclRuntime.hpp:68,87,121-124`                                                                                   |
| `processEpilogue` on the device path                       | **DROPPED FROM SCOPE (maintainer, 2026-08-25)** — it exists on the CPU for SIMD batch sizes and for tags splitting work into min-sample frames; neither applies to a device. No value, so not tracked |
| automatic transfer-block insertion at domain transitions   | **STILL OPEN** — no `HostToDevice`/`DeviceToHost` anywhere in `Graph.hpp`; the user still wires them by hand                                                                                          |

Also still open and unaffected by this sweep: F1 device-side mutation as a compile error (needs C++26) ·
`pr/0-buffer-leak` ready to open on main.

**Lesson worth keeping:** every item in this doc's OPEN/fix-list sections must be re-verified against the code
before being worked or quoted. Two separate ToDo lists this session carried items that the code had already closed.

## 29. B1 DESIGN DECIDED (Opus review, 2026-08-21) — Option 2, the aside channel

**Option 2 wins decisively, and my stated worry about it was WRONG.** I feared it would elide only the allocation
while a host ring copy remained. **There is no interior ring copy:** the producer writes IN PLACE into the ring the
consumer reads; `publish()`/`consume()` only move cursors. So if the producer simply does not write back, interior
host traffic is _zero_. The only residual is the wrap mirror (`CircularBuffer.hpp:378-400`), ~0.27 ns/sample against
the 38 being removed — already measured and ruled out as a target.

**The mechanism already exists.** `DeviceContextGLSL::pooledInputBuffer`/`pooledOutputBuffer` (`:159-160, 184-198`)
is already an aside channel: an SSBO pool keyed by `(blockKey, role)`, complete with `mirrorAllocationCount()`
(`:177-181`) as a guard against per-`work()` allocation. **B1 re-keys that map; it does not invent a mechanism.**
Option 1 would have reimplemented `workInternal` (settings — a frozen `fft_size` is silent wrong-size dispatch —
plus tag validity, error surfacing, message pump) _and_ hand-driven exported boundary ports that only alias member
ports. Same win, far more surface. Rejected.

**Transfer accounting, m-member GLSL chain, per `Domain::work()`:** today each member does H2D(n)+D2H(n) = **2m**.
One token per chain ⇒ one H2D at the head + one D2H at the tail = **2**. **2(m−1) transfers elided**; m=2 is 4→2.
Two more savings ride along: the hatch does not even use the existing pool (it allocates per call, `fft.hpp:199,206`)
and it stages a pure-waste `ranges::copy(inSpan→outSpan)` (`fft.hpp:192-196`) that must go, or it muddies the number.

**Validity rule (the Domain is single-threaded, topological, synchronous, so producer always precedes consumer):**
Domain bumps `_epoch` at the top of `work()`; the producer stamps `{epoch, samples}` after its kernel — the stamp IS
the record that it skipped its D2H; the consumer skips its H2D only if `stamp.epoch == currentEpoch && stamp.samples
== its own computed total`. **A mismatch is `work::Status::ERROR`, never a silent fallback**: each block computes
`total` independently from `min(inSpan.size(), outSpan.size())/N*N`, so tight downstream room makes the consumer
under-consume, and the producer has already skipped its write-back — no host fallback can reconstruct the remainder.
Invalidated by: new epoch · producer produced 0 / non-OK / did not run · `settingsChanged` altering required bytes ·
stop/start or quiescence · any reallocation.

**Requirement (b) is decided at BUILD time over the chain, not per edge.** Partition the inner graph into maximal
LINEAR chains (each source port exactly one outgoing edge, each destination exactly one incoming) and allocate one
token per chain. Because an in-place FFT deliberately aliases a chain's edges onto ONE token, the single-consumer
test must run over the CHAIN — per-edge it would read a 3-member chain as two innocent single-consumer edges while
one SSBO has two readers. Sequential readers along a chain are safe (one thread, topological order). True fan-out
ends a chain; those members revert to the host path. No per-`work()` check, no WebGPU versioning needed.
**Latent bug this exposes in my own code:** `makeDomain`'s `claimedOutputs` is a `std::set<PortKey>`, so a
fan-out duplicate collapses silently — it must COUNT, not insert, or fan-out is invisible.

**MVP:** GLSL only · forward only (`inverse=true` does host conjugate passes that touch the spans and defeat the
elision — refuse explicitly) · equal `fft_size` · one interior edge · tokens registered at FIXED size in
`startDispatch()` and never grown mid-call (the current pool grows on demand, and a grow between producer and
consumer hands the consumer a different token). Measured by transfer counters in the `mirrorAllocationCount()`
style: **assert 2 transfers per call, not 4.**
Deliberately uncovered: WebGPU (same hatch shape, second) · fan-out > 1 · inverse · quantum planning · tags (they
keep riding the host ring untouched) · cycles (refuse: `topologicalOrder` degrades to emplacement order on a cycle,
`DeviceDomain.hpp:134-138`, which breaks producer-before-consumer) · mixed CPU/GPU members · the two boundary
transfers.

**Highest risk + the test that catches it:** the producer skips its D2H, the consumer reads the host ring anyway
(partial drain, fallback, or a path ignoring the stamp) and silently transforms **stale bytes into plausible-looking
FFT output**. Test: poison the interior ring with a NaN sentinel after `startDispatch()`, drive known input through
a _constrained_ sink that forces the partial drain, and assert the result is either bit-exact against the CPU
reference or a loud `work::Status::ERROR` — never plausible wrong floats. One test, both failure modes.

## 30. A4 ANSWERED — the ~7x splits into 1.2-1.4x framework and 5x spectrum-mode work (2026-08-21)

`blocks/fourier/benchmarks/bm_fft_stream_graph.cpp` (new): `CountingSource<complex<float>>` -> `FFT<float>` (STREAM
mode, complex->complex — the same work the algorithm row does) -> `CountingSink<complex<float>>`, in a real graph
under a scheduler. gcc15 Release `-O3`, `-Werror` clean, best of 3, ~8 Mi samples per run.

Conversion used (both legs share the 5·N·log₂N convention): **MS/s = ops_per_s / (5·log₂N)**.

| N     | algorithm SimdFFT | in-graph, stream mode | **framework factor** |
| ----- | ----------------- | --------------------- | -------------------- |
| 1024  | 282 MS/s          | 198 MS/s              | **1.42x**            |
| 4096  | 228 MS/s          | 185 MS/s              | **1.23x**            |
| 16384 | 243 MS/s          | 171 MS/s              | **1.42x**            |

**So the framework costs ~1.2-1.4x — roughly 20-40 % on top of the raw algorithm — and NOT 7x.** The earlier ~7x
figure (40 MS/s in-graph spectrum vs 282 MS/s algorithm) decomposes as:
`282 --(1.4x framework)--> 198 --(5.0x spectrum-mode work)--> 40`.
The dominant term is **not** the framework: it is the window + magnitude + phase (`atan2`) + per-transform `DataSet`
allocation that spectrum mode adds. **The §24.1 caveat is now closed — quote 1.2-1.4x for framework overhead.**

**Two things the agent added beyond the brief, both worth keeping:**

- **Per-run sample count raised 1 Mi -> 8 Mi, disclosed.** At 1 Mi the wall time (5-15 ms) sat too close to
  `driveUntil`'s 200 µs poll granularity: two full runs reordered the three `fft_size` rows against each other by up
  to 45 %, so the sweep could not have said anything about N-dependence. At 8 Mi the ordering is stable and total
  runtime is ~2.2-2.5 s.
- **A control scenario** (`CountingSource -> Copy -> CountingSink`, no FFT): 260-346 MS/s. `CountingSource<T>` only
  gets a SIMD path for arithmetic `T`, and `complex<float>` falls back to scalar `processOne`, so without this
  control a slow FFT row could equally mean "the source cannot feed it faster". Every FFT row sits clearly below the
  control ceiling, so the measurement is FFT-bound, not source-bound.

### 30.1 A1 DONE — an interior edge really is device memory (2026-08-21)

`qa_DeviceDomain` now: **45 asserts / 5 tests** where no SYCL is compiled in (the hardware test skips, loudly) and
**56 / 5** under acpp — the delta of exactly 11 equals the new test's `expect()` count, which is the proof it ran
rather than silently skipping. Committed as `7a7eae52`; green on gcc15, clang20 and acpp.
The assertion that matters is that memory allocated from the interior edge's _resolved_ resource is dereferenceable
by the resolved context; checking merely that the resource differs from the default would have passed on host memory.
**Discovery:** on this machine the interior edge resolves to **`CudaVmmMemoryResource`** (2 MiB granule), i.e. the
device-only resource from PR-F — stronger than USM. See G22 for the landmine that goes with it.

## 31. B1 IMPLEMENTED AND MEASURED (2026-08-21) — 4 transfers per dispatch become 2

Commits: `5bff3249` (step 1, pooling + direct upload) and `08326df9` (step 2, the domain binding).

**The elision is proven by measurement, not by a green tick.** Two chained `FFT<float>` members, both
`compute_domain="gpu:glsl"`, in one `Domain`, driven 16 dispatches:

| configuration         | host transfers | formula            |
| --------------------- | -------------- | ------------------ |
| bound chain (shipped) | **34**         | `2·dispatches + 2` |
| binding forced off    | **66**         | `4·dispatches + 2` |

The `+2` is `GlslFFT::init()` uploading its twiddle table once per fresh block. Both numbers were _predicted from
the instrumented dispatch cadence and then measured exactly_. I reproduced the break independently: forcing
`needsUpload/needsDownload` true takes it to 66 and fails the test with `66 <= 34`; restoring returns 55/3 green.

**What shipped:**

- `DeviceContext::DomainBinding{chainId, needsUpload, needsDownload}` + `beginDomainEpoch` / `stampDomainChain` /
  `domainChainHolds`. The binding carries only identity and role — a domain cannot know a member's element type, so
  the member claims the chain buffer at the size it needs.
- `DeviceContextGLSL::pooledChainBuffer(chainId, bytes)` (chain-keyed, beside the existing block-keyed mirrors) and
  `hostTransferCount()` so the elision is assertable.
- `Domain::linearChains()` — maximal runs where each link is one producer to one consumer — and `bindChainBuffers()`,
  which binds a run **only if every member shares the same non-host `compute_domain`**.
- `processBulk_glsl` honours it: a non-head member skips its upload, a non-tail member skips its download and stamps.

**B2 is DONE, structurally and for free.** A run only extends across a 1:1 link, so a buffer with more than one
reader is never shared and never written in place — requirement (b) enforced at build time with no per-`work()`
check, exactly as the review argued.

**Two traps caught on the way, both of which would have shipped a vacuously-green test:**

1. The first allocation guard asserted the pooled count "did not grow" — but the OLD code path called `allocate()`
   directly, which never touches that counter, so it read 0 and passed while testing nothing. Fixed by also
   asserting the count is non-zero.
2. The existing domain tests build `Copy -> FFT` where only the FFT has a `compute_domain`, so no binding is ever
   registered and the new path never executed. Every one of them stayed green throughout the B1 work while proving
   nothing about it. The chain test had to build **two** members that both carry the domain.

**Still open in B-land:** B3's debug-fill flag. Skip-fill is now the de-facto behaviour — the producer no longer
writes the interior host ring, so it holds stale bytes — but the flag that would let a tap or bisect run see real
data there is not implemented. WebGPU is untouched (its context has no pool at all); the same hatch shape ports over.

## 32. A6 — WebGPU code path DONE, browser Domain run FAILS (2026-08-22). §22.6's risk is now VERIFIED, not theoretical.

Commits `51dac107` (B3) and `05f16c25` (WebGPU chain buffer). The GLSL and WebGPU hatches now share one treatment,
and `pooledChainBuffer` + `hostTransferCount` moved to the `DeviceContext` base so both backends and both stubs get
them — which also removed the `#if GR_DEVICE_HAS_GL_COMPUTE` guard the GLSL test previously needed.

**The browser result, on real hardware (`vendor=nvidia architecture=ampere`, headed Chrome):**

| phase                                  | result                                   |
| -------------------------------------- | ---------------------------------------- |
| standalone `processBulk_webgpu` sweep  | **PASS** (max rel err 1.6e-07)           |
| single-block graph, `gpu:webgpu`       | **PASS**                                 |
| **two FFTs chained inside a `Domain`** | **`Aborted(RuntimeError: unreachable)`** |

Both members dispatch — two "below recommended quantum" warnings appear — and then the WASM runtime traps. Nothing
downstream of the second dispatch runs, so no result line is printed.

**This is exactly what §22.6 warned about and left UNVERIFIED:** _"Asyncify does NOT dissolve — it moves. In the
domain model the submit+wait sits mid-PARENT-work()."_ A single block's WebGPU download suspends via Asyncify from
inside `traverseBlockListOnce -> block->work()` and that works (the graph phase passes). Put one more frame on the
stack — `Domain::work() -> member->work()` — and the unwind traps. The harness links `-sASYNCIFY=1` with no
`ASYNCIFY_ADD`/`ASYNCIFY_ONLY` list, so instrumentation is whatever the static call-graph analysis infers.

**Not diagnosed further, and deliberately not committed.** The harness change that reproduces it is preserved at
`<scratchpad>/webgpu_domain_phase.patch` (124 lines against `tools/webgpu/fft_graph/main.cpp`); it was reverted
because with it applied the whole browser harness exits 4, turning a green check red. Re-apply it to reproduce in
one run. **The harness is verified green without it** (`GR4_FFT_GRAPH_RESULT: PASS`, rc=0) _with_ the new WebGPU
hatch in place, so the hatch change itself is sound — it is the Domain frame that breaks.

**Candidate causes, in the order worth testing:** (1) Asyncify not instrumenting the `Domain::work()` frame —
cheapest probe is an `ASYNCIFY_ADD` entry for it, or `-sASYNCIFY_ADVISE=1` to see what the toolchain thinks can
suspend; (2) `Domain::work()` is `noexcept`, so anything that would propagate out becomes `std::terminate`, which
also presents as `unreachable` — dropping `noexcept` temporarily would separate the two; (3) chain-buffer lifetime
across the two harness runs (host then webgpu) sharing one process-global context.

**SUPERSEDED — FIXED, see §34.**

## 33. A3 DONE + FINAL STATE OF THE A/B BATCH (2026-08-22)

**A3: `build-ci-clang20-debug` now has `GR_ENABLE_GL_COMPUTE=ON`.** More than a flag flip — GLSL **compiles clean
under clang20/libc++ `-Werror`** (it had only ever been built by gcc15) and **actually creates a GL context there**:
`NVIDIA GeForce RTX 3070/PCIe/SSE2, 4.3.0 NVIDIA 610.57.04`. So the GLSL suite runs rather than skipping, and the
device path is now gated by both front ends. `qa_DeviceContext` there: 24/5 GLSL + 2157/8.

**Verified final state, all four configurations, HEAD `05f16c25` (69 commits above `origin/main`):**

| config                   | qa_FFTDomain    | qa_DeviceDomain | qa_DispatchGroup |
| ------------------------ | --------------- | --------------- | ---------------- |
| gcc15 release, GL ON     | **73/4** + 29/3 | —               | —                |
| clang20 debug, GL ON     | **73/4** + 29/3 | 45/5            | 24/2             |
| gcc15 debug (no backend) | —               | 45/5            | 24/2             |
| acpp (SYCL)              | —               | **56/5**        | —                |

Identical GLSL counts on both compilers is the point of A3.

**Batch scorecard against the request "A, B and the simple fixes":**

| item                                       | state                                                                      |
| ------------------------------------------ | -------------------------------------------------------------------------- |
| A1 SYCL domain residency on hardware       | **DONE** `7a7eae52`                                                        |
| A2 GL memory barrier                       | **was already done** — the ToDo entry was stale                            |
| A3 two-compiler gate for GLSL              | **DONE** (build config; GLSL now clang20-verified)                         |
| A4 stream-mode in-graph FFT                | **DONE** `bea62a8c` — framework cost is 1.2-1.4x, not 7x                   |
| A5 idle-card benchmark re-run              | **DONE 2026-08-22** — reproduces the reference, see §35                    |
| A6 WebGPU domain leg                       | **PARTIAL** — code path done `05f16c25`, browser run aborts, see §32       |
| B1 interior device buffers                 | **DONE** `5bff3249` + `08326df9` — 4 transfers/dispatch become 2, measured |
| B2 fan-out / WAR rule                      | **DONE** — structural, free, enforced at build time by chain partitioning  |
| B3 skip-fill + debug-fill flag             | **DONE** `51dac107`                                                        |
| "simple fixes" (F2/F9/F10/DeviceSpans doc) | **were already done** — 5 of 7 stale entries, see §D re-verification       |

**The two items the sweep left open** were `processEpilogue` on the device path (since DROPPED FROM SCOPE, 2026-08-25 — 0 references in
`ExecutionStrategy`) and automatic transfer-block insertion at domain transitions (nothing in `Graph.hpp`).

## 34. THE ASYNCIFY ABORT — ROOT-CAUSED AND FIXED (2026-08-22). A6 IS COMPLETE.

**Root cause: `ASYNCIFY_STACK_SIZE`, nothing else.** Asyncify saves the whole suspended call stack into a fixed
buffer that defaults to **4 KiB**. A domain adds frames (`BlockWrapper<Domain>::work` -> `Domain::work` ->
`member->work` -> `workInternal` -> `dispatchProcessing` -> `dispatch` -> `processBulk_webgpu` -> `download`), the
saved stack no longer fits, and the runtime traps as `unreachable`. **Fix: `-sASYNCIFY_STACK_SIZE=65536`** in the
harness link options (`tools/webgpu/fft_graph/CMakeLists.txt`), commit `0d494ae3`.

**All three of my ranked hypotheses were WRONG** — it was not Asyncify instrumentation of the `Domain::work()` frame,
not `noexcept` turning something into `std::terminate`, and not chain-buffer lifetime. Two cheap empirical steps beat
all of them:

1. **A one-member domain control** (one member forms no chain, so nothing is bound) **aborted identically** — which
   ruled the entire elision mechanism innocent before any of it was investigated.
2. **Letting the error surface**: Emscripten's own message says _"'unreachable' may be due to ASYNCIFY_STACK_SIZE
   not being large enough (try increasing it)"_. The first run had hidden it behind the harness's own FAIL line.
   **Causally confirmed both ways:** remove the setting and the abort returns (rc=4); restore it and the whole harness
   passes (rc=0).

**A6 RESULT — a compute domain now runs in a browser on real hardware** (`vendor=nvidia architecture=ampere`,
headed Chrome), and the elision is measured _in the browser_:

| check                                            | value                                              |
| ------------------------------------------------ | -------------------------------------------------- |
| host transfers, bound 2-member chain             | **4**                                              |
| ceiling if unbound                               | 16                                                 |
| spectrum vs a host run of the same 2-stage chain | **5.2e-07** relative                               |
| stability                                        | 3 consecutive runs, `1.21e-07` each, deterministic |

Transfer arithmetic checks out: the whole 4096-sample run is ONE dispatch of 4 batches, so a bound chain costs
1 upload + 1 download + 2 one-off twiddle uploads = 4; the one-member control costs 2 + 1 = 3, as measured.

**One flake seen and now made diagnosable:** a single run reported `max_rel_error=1.000000e+00`, which is exactly
what an **all-zero reference** produces — the _host_ reference run, not the device path. The harness now refuses to
report a relative error when either side is all zeros and says so instead, because a 1.0 reads as a numeric
mismatch when it actually means the comparison had nothing in it. Not reproduced in 4 subsequent runs; if it
returns, the message now names it.

**Lesson worth keeping:** when a device path aborts, run the _degenerate_ configuration first (one member, no
chain, nothing bound). It costs one build and either exonerates or convicts the whole mechanism in a single run.

## 35. A5 DONE — the idle-card sweep REPRODUCES the recorded reference (2026-08-22)

Card **idle and pinned at 1500 MHz** (10 % util, throttle reason = GpuIdle, 3.0 GB resident) — the protocol
condition, and the same pinning the recorded reference used. Best of 3 on both legs: native **105 PASS ×3**,
browser **64 PASS ×3**, adapter `WebGPU: (ampere)` every run.

**N=4096, recorded reference vs this run:**

| batch | backend      | recorded | now       | ratio |
| ----- | ------------ | -------- | --------- | ----- |
| x1    | SimdFFT      | 16.5G    | 15.4G     | 0.93x |
| x1    | SYCL:GPU     | 4.1G     | 4.7G      | 1.15x |
| x1    | GLSL:GPU     | 1.6G     | 1.7G      | 1.06x |
| x16   | SYCL:GPU     | 41.0G    | 43.6G     | 1.06x |
| x16   | GLSL:GPU     | 15.7G    | 16.5G     | 1.05x |
| x128  | SimdFFT      | 15.3G    | 15.4G     | 1.01x |
| x128  | **SYCL:GPU** | 76.3G    | **79.4G** | 1.04x |
| x128  | GLSL:GPU     | 14.0G    | 15.1G     | 1.08x |

Browser WebGPU: x16/N=8192 1.08x · x128/N=32768 0.94x · x1/N=65536 0.93x · x1/N=1024 1.36x (29M vs 21M — the
noisiest cell in the whole sweep, where a fixed per-dispatch cost dominates the arithmetic).

**Every GPU figure is within 8 % of the record and most sit slightly above it.** The 2026-08-20 contended sweep is
now superseded and kept only as evidence of what contention costs (GPU rows were 2.5-40x low while CPU rows held —
the internal control working exactly as intended). The **memory reference
`reference_fft_backend_benchmark_table` now carries this idle-card table as authoritative.**

Peak figures worth quoting from this sweep: **SYCL:GPU 86.7G** (x16/N=65536) and **80.2G** (x128/N=8192);
in-browser WebGPU peaks **19.1G** (x16/N=16384).

**Two oddities present in BOTH sweeps, so characteristic and not noise:** in-browser SimdFFT collapses to ~1.8G at
x128/N=65536 against ~5G everywhere else (largest size x largest batch, WASM heap pressure), and GLSL:GPU stays
erratic between neighbouring cells (x16: 16.5G at N=4096, 5.7G at N=8192).

## 36. MAINTAINER DECISIONS 2026-08-22, and what they changed

### 36.1 The CPU-fallback tag drop is NOT A BUG — the merge blocker dissolves

Maintainer: _"Shouldn't only the 'gr:' prefixed tags be forwarded? I'd like to slowly discourage doing tag handling
in blocks that implement only processOne(...) because of the performance penalty of checking for every sample
whether there is a tag, and rather have those blocks auto-forward the tags (those with 'gr:' or explicitly
enabled)."_
So the filtered `forwardInputTags` behaviour is **intended**, not a defect. `qa_DeviceSeam`'s custom `"kind"` key is
outside the forwarded set by design, and hand-forwarding it inside `processBulk_sycl` is the pattern being
discouraged. **One of the two self-declared merge blockers is therefore closed by decision, not by code.**
Follow-up: the test encodes the discouraged pattern and should move to a `gr:` key or the auto-forward supplement.

### 36.2 Tags on device — the investigation INVERTS the question

**Tag values already reach auto-parallel kernels today, with zero per-sample cost.** Verified chain:
`applyInputTagsFromPorts` -> `settings().autoUpdate()` (`Block.hpp:1334`) -> `autoUpdateImpl` stages, gated on
`autoUpdateParameters().contains(key)` (`Settings.hpp:330-336`) -> `applyChangedSettings` bumps `_settingsEpoch`
(`Block.hpp:1354,1359`) -> the device mirror refreshes BECAUSE the epoch moved (`ExecutionStrategy.hpp:361-369`) ->
the kernel reads the fresh member. Scope, honestly: `gr:` keys that are ALSO reflected auto-update settings on that
block — which is most of the forwarded set, and exactly what §36.1 says should be forwarded.

- **SYCL:** the slab the maintainer meant is `DeviceLogSlab` (`DeviceLog.hpp:45-100`) over `ValueMapView::formatAt`
  — but it is **kernel-write + host-read only**; kernel-side READ is plausible-but-unproven (the `get_if` path hits
  a fixed-size memcpy, the G19 shape, never audited on device) and **publish into a ring is impossible** (`claim()`'s
  unordered `fetch_add` cannot produce index-ordered insertion).
- **A pointer route exists but is BLOCKED BY A LATENT BUG:** `Tag` is already trivially copyable for USM by-value
  transport (`Tag.hpp:84-93`) and the tag axis is forced `Access::Shared`, but `kBlobAlignment = 16`
  (`ValueMap.hpp:267`) while `serialiseBlob` packs at 8-byte at best (`ChunkBuffer.hpp:153,160`) and `makeView`
  casts without validating. x86 tolerates it; a GPU generally will not. **Gate for that route: assert
  `tag.map._blob % 16 == 0` on a device edge — it currently will not hold.**
- **GLSL/WebGPU: read-only, expert hatch only.** `GlslFFT` already binds a third read-only SSBO for its twiddle
  table (`GlslFFT.hpp:120,166-168`) — exactly the shape a tag buffer wants. The framework-generated tier is not
  extensible (two buffers hardcoded, `ShaderFragment.hpp:46-47`). **Publish is impossible on both** — no shader
  atomics, and opaque contexts refuse `allocateShared`.
- **Recommendation:** ship the settings route as the answer and document it; add a read-only POD occurrence record
  only for the residual (tag _occurrence_ + sample index, which settings never give you), SYCL first; do NOT name
  the accessor `rawTags()` — that member's ABSENCE is load-bearing for duck-typed feature tests
  (`DeviceSpans.hpp:18-20`).
- **BIGGEST RISK, unmeasured:** a matching tag bumps `_settingsEpoch` **even when the value is unchanged**, and an
  epoch move relocates `sizeof(TBlock)` to the device (`ExecutionStrategy.hpp:365-368`). Cheapest experiment: count
  `relocateBlockToDevice` on a tag-bearing vs tag-free stream, same block. Candidate one-line fix if it scales with
  tag rate: gate the epoch bump on `!appliedParameters.empty()`.

### 36.3 Closing items — two done, one blocked with the blocker root-caused

- **`ASYNCIFY_STACK_SIZE` now in ALL browser targets** (`3c3ce743`), not just the one where the abort was found.
- **Cycle guard** (`57ec38f2`): a domain that cannot be fully ordered shares no buffers. Tested WITH a positive
  control on the same two blocks, since "no chains" is also what a domain that never forms chains reports.
  52/6 gcc15 + clang20, 63/6 acpp.
- ~~**SYCL elision measurement BLOCKED**~~ — **ANSWERED, see §37.** The root cause below was correct and the fix
  was one flag. Original finding: It needs a build with the block registry
  ON _and_ AdaptiveCpp; none exists. Reason, finally pinned down: `CMakeLists.txt:808` requires
  `find_package(Python3 3.12 COMPONENTS Interpreter Development NumPy)` and **the default interpreter is 3.13, which
  has no dev headers** — while **3.12 has both headers and NumPy 2.3.3**. So the fix is one flag:
  `-DPython3_EXECUTABLE=/usr/bin/python3.12`. **This also unblocks `qa_FFTPerformance`, which has never run in this
  branch's history for exactly this reason.**
  Note `gr::testing::Copy` is NOT device-eligible (it warns "no device path available" and runs on the CPU), so the
  measurement needs the FFT — hence the registry.

### 36.4 `b0fcf8e0` message amended (maintainer chose "amend only")

Rebased non-interactively (`git commit --amend` on a detached checkout + `rebase --onto`, since interactive rebase
is unavailable here). Backup at `backup/pre-msg-amend-20260822`. **Verified: tree IDENTICAL to the backup, same 70
commits, and no "about +10%" remains anywhere in the branch log.** The message now says the difference does not
resolve above the run-to-run spread, with the sub-graph faster than flat as often as not.

## 37. THE acpp REGISTRY BUILD — ⚠️ THIS SECTION'S HEADLINE CLAIM WAS WRONG, see §47 (2026-08-22)

> **Correction:** Python was never required, and there was never a blocker. `GR_ENABLE_BLOCK_REGISTRY` defaults to
> **ON**, so the CI acpp lane has always had it — the local `build-acpp` simply had it explicitly configured OFF.
> The `-DPython3_EXECUTABLE=` flag in the command below was a coincidental passenger, not the fix. Everything below
> about the SUITES (qa_FFTPerformance 21/11, qa_FFTDeviceSycl, qa_SyclFFTSweep 84/84, and the SYCL elision figures)
> stands; only the explanation of why they had not run locally was wrong.

`build-acpp-registry`: acpp Release, GL ON, CUDA-VMM ON, **block registry ON**, `-DPython3_EXECUTABLE=/usr/bin/python3.12`.
**Configure succeeded in 32 s** — `Found Python3: /usr/bin/python3.12 ... components: Interpreter Development NumPy`
and _"Is block registry enabled? ... ON"_.
The blocker was never AdaptiveCpp and never the block-lib generator: `CMakeLists.txt:808` asks for
`Python3 3.12 COMPONENTS Interpreter Development NumPy`, the machine's default interpreter is **3.13 with no dev
headers**, and **3.12 has both headers and NumPy**. One flag.

### 37.1 `qa_FFTPerformance` RAN FOR THE FIRST TIME — 21 asserts / 11 tests, rc=0

It is the only place the `requires std::same_as<T, float>` constraint on the stream SYCL hatch is exercised, and it
had never executed in this branch's history. It also answers the SYCL interior-elision question directly:

```
   pairs  transforms  interior/total edges   rate [MS/s]   per-sample [ns]   marginal per pair [ns]
       1           2           1/5                10.2              98.4                    98.4
       2           4           3/7                10.2              98.1                    -0.3
       3           6           5/9                10.1              99.3                     1.2
```

**Adding a device pair — and with it two more interior edges — costs essentially nothing per sample** (-0.3 and
+1.2 ns against a 98.4 ns baseline; 1 interior edge to 5 changes the rate by 1 %). By the benchmark's own reading
guide that means **the two fixed host-boundary edges dominate and the interior edges are effectively free** — which
is the SYCL elision, measured. It arrives as a marginal-cost curve rather than a transfer count, which is arguably
the better evidence: it is end-to-end and needs no instrumentation.
**So all three closing items are now answered**, and the "SYCL is the one backend with a mechanism and no evidence"
gap from §36.3 is closed.

### 37.2 Two more never-run suites now pass

- **`qa_FFTDeviceSycl`** rc=0 — exercises both devices in one run: `NVIDIA GeForce RTX 3070 (cpu=0 gpu=1)` and
  `AdaptiveCpp OpenMP host device (cpu=1 gpu=0)`, inverse-batch error 3.8e-07 on each.
- **`qa_SyclFFTSweep`** rc=0 — **84/84 cells pass** at tol 1e-04 across the N x workgroup x batch sweep, forward
  L2 ~1.6e-07 and round-trip L2 ~2.6e-07 at N=65536.

**Keep `build-acpp-registry`.** It is the only configuration that can run the acpp + registry suites, and rebuilding
it costs ~4 minutes for `qa_FFTPerformance` alone. `build-acpp` (registry OFF) stays as the fast device-suite dir.

## 38. SUB-GRAPH AUTO-FORMATION — DROPPED FROM SCOPE (maintainer, 2026-08-25)

Automatic formation of a sub-graph from a flat graph's `compute_domain` annotations was designed here and
deferred. It is now **dropped from the requirements and the ToDos outright**: it is a _helper_, it needs no core
adjustment, and it can be a later addition or a separate PR if anyone wants it. Tracking it further only adds
load and noise, so the ~50 lines of deferred design that stood here have been removed. Explicit membership via
`makeSubGraph` / `makeDeviceSubGraph` is the supported way to form a group, and that is a complete story on its
own.

## 39. GOAL 2 IS BROKEN FOR THE PROJECT'S OWN RECOMMENDED BLOCK FORM (found 2026-08-22, measured)

Goal 2 reads: _"`const noexcept processOne` auto-parallelises onto any backend (**same gate as SIMD**)."_
**It does not hold for a templated `processOne` — which IS the SIMD gate.** Measured with a probe that instantiates
the concepts directly:

| block form                                | HasConstProcessOne | HasNoexceptProcessOne | AutoParallelisable | DeviceEligible |
| ----------------------------------------- | ------------------ | --------------------- | ------------------ | -------------- |
| **templated** `processOne` (SIMD-generic) | true               | **false**             | **false**          | **false**      |
| plain `processOne`                        | true               | true                  | true               | true           |
| `gr::testing::Copy<float>`                | true               | **false**             | **false**          | **false**      |

**Cause, `BlockTraits.hpp:396`:**
`concept HasNoexceptProcessOneFunction = HasProcessOneFunction<D> && IsNoexceptMemberFunction<decltype(&D::processOne)>;`
`decltype(&D::processOne)` is **ill-formed for a template**. The file ALREADY knows this: six lines below, the
merge-API concept `HasNoexceptProcessFunction` (`:410-416`) carries the escape
`(!requires { &Derived::processOne; } || ...)` and a comment saying templated process functions "can't be probed via
member-function-pointer ... for those cases we fall back to trust the declaration". **That escape was never applied
to the auto-parallel concept.**

**Consequences, in order of severity:**

1. A block written in the form the style guide encourages — `template<meta::t_or_simd<T> V> auto processOne(V) const
noexcept` — **silently never reaches a device**. It is not refused loudly; it warns and runs on the CPU.
2. **The warning actively misleads**: _"it needs a const noexcept processOne"_ is printed to an author who wrote
   exactly that (`Block.hpp:1944`).
3. This is why `gr::testing::Copy` fell back to the CPU when used as a device chain member (§36.3), which is what
   made the SYCL elision unmeasurable that way.
4. It plausibly explains why nobody noticed: every in-tree device block reaches the device through an explicit
   hatch (`processBulk_sycl`/`_glsl`/`_webgpu`) or `shaderFragment`, not through `AutoParallelisable`. **The
   auto-parallel path — goal 2, the headline promise — may have no real users at all.**

**Candidate one-line fix**, mirroring the pattern already in the same file:
`concept HasNoexceptProcessOneFunction = HasProcessOneFunction<D> && (!requires { &D::processOne; } || IsNoexceptMemberFunction<decltype(&D::processOne)>);`
**Not applied** — it changes which blocks dispatch to a device, and the "trust the declaration" fallback means a
templated _non_-noexcept `processOne` would then be accepted. That is a maintainer decision, and the same trade the
sibling concept already takes deliberately.

**Verify any fix with the probe, not by reading:** instantiate `AutoParallelisable` for a templated and a plain
block side by side; the table above is the expected-fail baseline.

## 40. INDEPENDENT RE-REVIEW (2026-08-22) — ALL ITEMS NOW CLOSED (see 40.7); the two blockers were introduced by this session's own work

An adversarial review, run against the code with the working doc treated as a claim log. It independently reproduced
the goal-2 defect (§39) from a compiled probe, and found several things I did not. **The two blockers below are
mine, from the Domain work, and neither is caught by any existing test.**

### 40.1 ~~BLOCKER~~ FIXED (`63e32276`) — per-Domain state lived on the process-global DeviceContext, unsynchronised

`_domainEpoch` is a plain `++` (`DeviceContext.hpp:108`); `_domainStamps` and `_chainMirrors` are plain
`unordered_map`s written from the dispatch path (`fft.hpp:227,304`, `DeviceContext.hpp:126`, where `operator[]` may
rehash). Only `_served` is atomic. **Two `Domain` blocks in one graph under `ExecutionPolicy::multiThreaded` run
`work()` on two pool threads against that shared state.** Every test uses `externalStep` with a single Domain, so
nothing exercises it. This sits in the backend-independent base, so it is not GL thread-affinity in disguise — it
bites SYCL identically, where queues are thread-safe and there is no excuse.
**Resolution:** the epoch is now kept **per domain** (a shared counter would let one domain invalidate another's
stamps and fail its chains for no reason), stopping a domain drops **only its own** bindings, and a mutex covers the
maps — touched about once per member per `work()` against a device dispatch, so the cost is noise.

### 40.2 ~~BLOCKER~~ FIXED (`085e98e7`) — no tag crossed a domain boundary in ANY test

`grep '_tags|publishTag|rawTags'` over `qa_DispatchGroup`, `qa_DeviceDomain` and `qa_FFTDomain` returns nothing;
`qa_FFTDomain` instantiates a `TagSink` and reads only `_samples`. **CLAUDE.md §7 makes tag-propagation coverage
mandatory**, and a Domain drives its members through a private inner graph across two `exportPort` hops — exactly
where tags would be lost. Untested in the one place most likely to break.
**Resolution:** `qa_DeviceDomain` now drives a `gr:`-prefixed tag from a `TagSource`, through a Domain of two
members, into a `TagSink`, and asserts arrival **and index**. It passes — a domain does not lose tags. The key is
`gr:`-prefixed deliberately: that is the auto-forwarded set (§36.1), and a test using a custom key would have
demonstrated the discouraged pattern instead.

### 40.3 ~~DESIGN FLAW~~ FIXED (`63e32276`) — chain buffers were keyed by a per-Domain index on a global context

`pooledChainBuffer` keys on `chainId` alone (`DeviceContext.hpp:125`), a **0-based per-Domain index**. Every
Domain's first chain is `chainId 0`. It survives today only because `clearDomainBindings()` wipes the whole map
whenever a Domain adopts the context (`DeviceDomain.hpp:205`), degrading a collision into a lost binding rather than
a shared buffer. **Add per-Domain retention, second-Domain support, or remove that wipe, and two Domains write the
same device buffer and produce plausible wrong numbers with no error.** Key by `(context, domain identity, chainId)`
before anything else touches this.
**Resolution:** chain ids now come from an atomic counter **on the context** (`DeviceContext::nextChainId()`), so
they are unique across every domain sharing that backend and cannot collide by construction. The binding also
carries the owning domain, used purely as a key and never dereferenced.

### 40.4 API sharp edge — adding a block RENAMES an existing port

`detail::boundaryName` (`DeviceDomain.hpp:302`) returns `in` for one boundary input and `in0` for two. **Adding a
second block to a sub-graph renames the first port**, breaking every parent edge, persisted description or
hardcoded string that referenced it. My "reads like an ordinary block" convenience is the cause.

### 40.5 Weak tests — mine, and each would pass for the wrong reason

- **`runChainedFftsInDomain` pins `max_work_items` to `kFftSize`** (`qa_FFTDomain.cpp:115`), which GUARANTEES
  producer and consumer compute the same `total` — precisely the condition the `domainChainHolds` ->
  `work::Status::ERROR` path (`fft.hpp:205-211`) exists to catch. **The only test of the bound-chain path excludes
  its own failure mode.** The pin was added for a deterministic transfer count and is otherwise justified; the gap
  is that no second test covers the mismatch.
- **The debug-fill test never reads the interior ring** (`qa_FFTDomain.cpp:318-348`) — it asserts the transfer count
  rose and the output did not change, which is consistent with the ring still holding stale bytes.
- ~~**`_hasCycle` is never reset** — sticky across stop/start.~~ **FIXED** (`63e32276`): reset in `stopDispatch()`.
- ~~**A stale comment I created:** `qa_FFTDomain.cpp:257` claims the no-GL stub lacks the two counters.~~
  **FIXED** (`a1146daa`): the guard is about the GLSL-specific cast, not about absent members.
- **`DeviceContextCpu` never bumps `_hostTransfers`**, so any future CPU transfer test passes vacuously.

### 40.6 Goal scores (independent, and harsher than mine)

`1` MET · `2` **NOT MET** (§39) · `3` MET · `4` PARTIAL (one `compute_domain` string, TWO parallel registries:
`DeviceContextRegistry` and `ComputeRegistry`, independently "served") · `5` **PARTIAL** · `6` PARTIAL (device
scenarios self-skip, so **a lane where GL stops being served goes green with zero device assertions** — CI-skippable,
not CI-testable) · (
nothing that runs) · `8` PARTIAL ("one seam" is eight members plus four accessors plus an inline branch; `Domain`
itself is genuinely composed).

**Goal 5, corrected — I got the mechanism wrong.** `DeviceContext.hpp` DOES reach every TU, via the unconditional
`#include <device/DeviceBlockShadow.hpp>` at `Block.hpp:33` (whose comment says it "pulls only DeviceContext.hpp,
which is backend-free"). I checked only the gated `ExecutionStrategy.hpp` include ten lines below and wrongly
concluded my additions could not reach a CPU-only build. **Measured cost of my additions, however, is nil:**
0 preprocessed lines (`<unordered_map>` already arrived by another path) and `sizeof(Block<P>)` unchanged at 2752.
The standing goal-5 costs are pre-existing: `DeviceContext.hpp` is ~51k preprocessed lines in every TU, and
`_computeDomainIsDevice` is branched on at runtime with zero backends compiled.

### 40.7 EVERY REVIEW ITEM IS NOW CLOSED (2026-08-22)

| #      | finding                                                                         | resolution                                                                       |
| ------ | ------------------------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| 40.1   | per-Domain state unsynchronised on a global context                             | `63e32276` — per-domain epoch, own-bindings-only teardown, mutex                 |
| 40.2   | no tag crossed a domain boundary in any test                                    | `085e98e7` — `gr:` tag asserted across both hops, index included                 |
| 40.3   | chain buffers keyed by a per-Domain index                                       | `63e32276` — ids from an atomic counter on the context                           |
| 40.4   | a second boundary port renamed the first                                        | `3b67a8db` — `<block name>:<port name>`, duplicates refused                      |
| 40.5   | `_hasCycle` sticky · stale stub comment · CPU counters · `disconnect_on_done`   | `63e32276`, `a1146daa`, `f01a7c62` (§44 — the override never worked and is gone) |
| 40.5   | bound-chain test excluded its own failure mode · fill test never read the ring  | `d5429a39` (§43)                                                                 |
| 39     | goal 2: templated `processOne` not device-eligible                              | `fcff158f` (§42.1) — call-noexcept probe, no regression                          |
| 40 API | every unclaimed port exported, no opt-out · two-backend group accepted silently | `4499c2d4` — `doNotExport`, and >1 device domain refused                         |
| goal 6 | device lanes could pass green having skipped everything                         | `2826358a` (§45)                                                                 |

**Nothing from the independent review remains open.** What is still open branch-wide is listed in the READ FIRST
block: ~~`processEpilogue` on the device path~~ (dropped 2026-08-25), ~~automatic transfer-block insertion~~ (dropped 2026-08-25, §38), F1 (needs C++26), the blob
alignment bug, `pr/0-buffer-leak`, and the PR re-cut.

## 41. ~~AGREED NEXT STEPS~~ — ALL DONE (2026-08-22)

Was: fix the goal-2 gate after illustrating it, fix the `boundaryName` renaming, and leave review items 3-7 open.
All three are closed — §42 (gate + naming), §43, §44, §45, §46, and the table in §40.7. Kept only as the record that
the gate change was illustrated and approved before it was applied, rather than taken unilaterally.

## 42. GOAL 2 AND THE PORT NAMING — BOTH FIXED (2026-08-22)

### 42.1 The goal-2 gate: option C, measured, no regression

Maintainer chose the call-based probe over the one-liner. `HasNoexceptProcessOneFunction` now asks whether the
processOne **call** is noexcept instead of whether its address can be taken:
`noexcept(declval<const TBlock&>().processOne(get<Is>(declval<const inputs&>())...))`.

| block form                 | before      | after        |
| -------------------------- | ----------- | ------------ |
| plain, noexcept            | eligible    | eligible     |
| plain, NOT noexcept        | refused     | refused      |
| **SIMD-generic, noexcept** | **refused** | **ELIGIBLE** |
| SIMD-generic, NOT noexcept | refused     | **refused**  |

The rejected one-liner ("trust the declaration when the address cannot be taken", as the sibling concept does)
would have made the last row eligible too — dispatching a possibly-throwing processOne to a GPU.
**Pinned as `static_assert`s in `qa_Block`** so a regression cannot compile.
**Regression swept:** qa_Block, qa_Graph, qa_Scheduler (344/31), qa_DynamicPort, qa_DeviceDomain, qa_DispatchGroup
on gcc15; qa_Block/qa_Scheduler/qa_DeviceDomain on clang20; qa_DeviceDomain + **qa_DeviceBlockStyles (65/8)** on
acpp — the last being the suite most sensitive to which blocks now dispatch.

### 42.2 Boundary ports are named `<block name>:<port name>`

Maintainer's scheme, and it is sound in both senses — checked, not assumed:

- **Technically:** `PortMetaInfo::name` is an owning `Annotated<std::string>` (so it copies — G20's dangling
  `string_view` hazard does not apply); nothing splits port names on `:`; connections serialise as a **4-element
  sequence** so the port name is its own scalar and cannot collide with YAML's key delimiter; `emplaceEdge` takes
  block and port separately.
- **Semantically:** the name depends only on the member, so adding a member never renames an existing port. It
  trades "adding a block renames a port" for "renaming a block renames its own port" — deliberate and predictable.

**The trap, and why makeDomain refuses:** `name` defaults to the TYPE name and is documented _"may not be unique ->
::unique_name"_ (`Block.hpp:754`). Two unnamed members of one type would export one name between them. Duplicates
are refused at construction with an error naming what to change, because two ports under one name surface as a
lookup resolving to neither — per G20, a graph that connects nothing and hangs to the ctest timeout.
The collision is NOT the common case: in a linear chain the boundaries are the head's `in` and the tail's `out`, so
the port names differ even when the block names do not. A test covers both the refusal and a named positive control.

**Verified:** qa_DeviceDomain 66/8 gcc15 + clang20, 77/8 acpp · qa_FFTDomain 73/4 + 29/3 with the GLSL elision
intact · browser domain still PASS on hardware.

## 43. THE TWO WEAK TESTS — FIXED (2026-08-22)

### 43.1 The mismatch guard is now actually exercised

The pinned test stays as it is: the pin is what makes its transfer count deterministic. A **second** test starves the
outgoing edge (`minBufferSize = 2*fft_size`) with **no pin**, so the tail can only take a smaller multiple of
`fft_size` than the head already left in the shared buffer.

**Instrumented to confirm it reaches the branch, rather than assumed: the guard fired 16 times.** Without that check
the test would have passed on "the starved edge simply never let anything through" — the same wrong-reason pass the
review was complaining about.

**Control: with the guard removed the test FAILS** (plausible wrong data appears); restored, 91 asserts / 5 tests.
The assertion is deliberately two-sided — nothing comes out, or what comes out is right; a wrong-but-plausible
spectrum is the failure it exists for.

### 43.2 The fill test now names the ring it claims to check

It asserted transfers went up and the output did not change — neither of which says the interior **host ring** was
written. Uploads and downloads are now counted separately (`hostUploadCount()` / `hostDownloadCount()`), because a
download IS `copyDeviceToHost(buffer, outSpan.data(), ...)`, i.e. the write into that ring. The test now asserts:

- `filledUploads == leanUploads` — filling must not add an upload, since the shared buffer already holds the data;
- every extra transfer is a download;
- `filledDownloads == 2 * leanDownloads` — a two-member chain downloads once per dispatch lean and twice filled.

**Verified:** qa_FFTDomain 91/5 + 29/3 gcc15-release with GL, rc=0 clang20, qa_DeviceDomain 77/8 acpp, WASM builds
clean (the counter split touched the WebGPU context too).

## 44. THE `disconnect_on_done` OVERRIDE NEVER WORKED (2026-08-22)

The review called it a silent overwrite. **It was worse: it silently did nothing.** Measured — `head`, `middle` and
`tail` all read `disconnect_on_done == true` before `makeDomain`, after it, and after `start()`. So
`member->settings().set({{"disconnect_on_done", false}})` + `applyStagedParameters()` through a **BlockModel** never
reaches the member's reflected field, and the line had been a no-op for its entire life while its comment called it
necessary ("both boundary peers are attached by the parent, so a member must not stop for want of a neighbour").

**Everything passes without it having ever taken effect** — every domain test, the tag-boundary test, the GLSL
elision, the in-browser run. So it is **removed** rather than repaired: deleting a no-op cannot change behaviour,
whereas making it work would change behaviour that nothing has ever needed changed. Verified after removal:
qa_DeviceDomain 73/9 gcc15 + clang20, 84/9 acpp, qa_FFTDomain 91/5 + 29/3 unchanged.

A test now pins the real contract — **a helper does not quietly change settings its caller chose** — so a future
attempt to reintroduce this has to come with a mechanism that demonstrably applies and an assertion on the value.

**Worth remembering as a trap:** `settings().set()` on a `BlockModel` stages a value that never lands on the concrete
block's field. The route that works is assigning the member directly on the concrete type
(`fft.disconnect_on_done = false`, as `bm_fft_subgraph` does), which a helper holding only `BlockModel*` cannot do.

Also fixed here: **`DeviceContextCpu` now counts its uploads and downloads** like every other backend. The base
class documents the counter as the way to tell whether an elision happened, and on the CPU context it never moved —
so a test asserting an elision there would have measured zero either way.

## 45. A DEVICE LANE CAN NO LONGER PASS GREEN HAVING SKIPPED EVERYTHING (2026-08-22)

The review's goal-6 finding: every device scenario self-skips, so a lane that stops serving its backend passes with
zero device assertions — the failure is indistinguishable from the success. **This branch has already been in that
state once**: the GL lane was green while covering nothing, because `GlComputeContext::init()` could not create a
context in the container at all and every scenario returned early (see the 2026-08-09 GL CI block).

**Mechanism:** `GR4_REQUIRE_DEVICE` names the domains a run must actually reach, comma-separated. All 11 skip sites
across `qa_FFTDomain` and `qa_DeviceDomain` now assert `!deviceDomainRequired(<domain>)` before skipping, so a
required-but-unserved domain FAILS instead. Helper is header-only in
`blocks/testing/include/gnuradio-4.0/testing/DeviceExpectation.hpp` — `gr-testing` is already linked by both test
directories. Matching is on whole comma-separated entries, so `gpu:sycl` does not satisfy a required `gpu:sycl:0`.

**It is derived from the build, not from CI.** `CMakeLists.txt` appends `GR4_REQUIRE_DEVICE=gpu:glsl` to the shared
`_GR_TEST_ENV` whenever `GR_HAS_GL_COMPUTE` is on, so a lane demands what it was configured for and cannot forget.
Verified attached in `build-ci-gcc15-release`, `build-ci-clang20-debug` and `build-acpp`; correctly ABSENT in
`build-gcc15-debug` (GL off).

**Deliberately NOT auto-required: SYCL.** The CI acpp lane pins `ACPP_TARGETS=omp`, so `gpu:sycl` is genuinely
unservable there and requiring it would fail a lane that is behaving correctly. The guard exists at those sites, so
a lane that DOES have a GPU can opt in by setting the variable.

**Proven both directions:**

| run                                            | result                                                   |
| ---------------------------------------------- | -------------------------------------------------------- |
| no requirement, no SYCL                        | skips, passes (unchanged for a laptop)                   |
| `GR4_REQUIRE_DEVICE=gpu:sycl`, no SYCL         | **FAILS, exit 255**                                      |
| `GR4_REQUIRE_DEVICE=gpu:cuda,gpu:hip`, no SYCL | passes — an unrelated entry does not trip it             |
| GL served + requirement (the real lane)        | passes with 91 device asserts, and green through `ctest` |

## 46. THE LAST TWO makeDomain ITEMS (2026-08-22)

**Export opt-out.** Exporting every unclaimed port stays the default — a port the parent cannot reach is one nothing
can feed or drain — but it was the only behaviour available. `makeDomain(graph, {"tail:out"})` now keeps named ports
private, using the same `<block name>:<port name>` labels the handle returns.

**A domain may hold at most ONE device `compute_domain`.** Two were previously accepted, after which one got its
buffers bound and the other fell back to the host path with nothing reporting it. Host members alongside a single
device remain legal — a chain stops where the domain changes, and that mix is exactly what the existing tests use.

**The positive control paid for itself immediately.** `compute_domain` defaults to
`gr::thread_pool::kDefaultIoPoolId` (`"default_io"`), **not** `"host"` or empty — so the first version of the rule
counted every ordinary block as its own device domain. Without the "host member + one device is accepted" half of
the test, it would have shipped refusing any domain containing an unannotated block, breaking `qa_FFTDomain`'s
`Copy -> FFT` outright. The rule now treats the two pool ids as host, matching `Graph::applyEdgeConnection`.

Verified: qa_DeviceDomain **84/11** gcc15 + clang20, **94/11** acpp; qa_FFTDomain 91/5 + 29/3 unchanged.

## 47. CORRECTION: PYTHON WAS NEVER NEEDED, AND acpp+REGISTRY NEEDS NO CI LANE (2026-08-22)

The maintainer asked why Python was being queried at all and what was missing. **The answer is: nothing was
missing, and I was wrong twice.** Both proven, not argued:

1. **The block registry does not need Python.** Configuring with `-DGR_ENABLE_BLOCK_REGISTRY=ON` and **no Python
   flag whatsoever** succeeds: CMake reports _"Could NOT find Python3 (missing: ... Development NumPy ...)"_ and
   _"Is block registry enabled? ... ON"_ in the same run, the block-lib generator runs 35 times, and
   `qa_FFTPerformance` builds and links. `CMakeLists.txt:807` says so outright — _"Python integration is optional —
   only enables `setup_test()` linking and the NumPy Python check"_ — and `PYTHON_AVAILABLE` gates only four places,
   none of them the registry.
2. **`REGISTRY_DEFAULT` is ON** (`CMakeLists.txt:170`, off only for MinSizeRel). The CI AdaptiveCpp lane passes no
   registry flag, so **it has always had the registry, and `qa_FFTPerformance`, `qa_FFTDeviceSycl` and
   `qa_SyclFFTSweep` have always run there.** The local `build-acpp` had it explicitly OFF — a build-directory
   setting I mistook for a constraint and then generalised to CI.

**So there is no acpp+registry CI lane to add.** The §37 claim that "a two-year-old blocker was one CMake flag" was
wrong on both halves: no blocker, and the flag was irrelevant. The suites and figures §37 reports are unaffected.

### 47.1 The browser lane — the gap is real, and the runner should be the toolchain's

The emcc CI lane runs its wasm tests under **node**, which has no WebGPU, so nothing in CI has ever exercised the
backend. A lane is added (`.github/workflows/ci.yml`, job `webgpu-browser`, YAML validated).

**It uses `emrun`, not a script of ours.** `emrun` ships with Emscripten — the toolchain any wasm build already
requires — so it adds no dependency; `tools/webgpu/fft_graph/CMakeLists.txt` now links with `--emrun` so the runtime
forwards the page's stdout and exit code. This is the answer to "no additional python script": our
`run_wasm_webgpu.py` exists because capturing a page's console over CDP needs a WebSocket client, and `emrun`
already solves that problem in the toolchain.

**Proven once, end to end:** `emrun --browser chrome --browser-args="--headless=new --enable-unsafe-webgpu
--use-angle=vulkan" --kill-exit harness.html` returned **rc=0 with STANDALONE / GRAPH_WEBGPU / DOMAIN_WEBGPU /
GR4_FFT_GRAPH all PASS**.

**⚠️ NOT reliably reproducible, and the lane is therefore UNVALIDATED.** A fresh build directory returns rc=159 with
no result line. Ruled out by test: the `--emrun` link flag (21 runtime hooks present in the js either way), the build
type, `--no-sandbox`, and port contention. Not isolated. **Treat the lane as needing a first run on a real runner
before it is trusted**; it fails visibly rather than silently, which is the acceptable failure mode for a new lane.

**Also confirmed, contradicting a tempting misread:** headless Chrome's WebGPU adapter really is SwiftShader
(`vendor=google architecture=swiftshader`). The `Selected adapter: NVIDIA GeForce RTX 3070 ... backend=Vulkan` line
that appears with `--use-angle=vulkan` is **ANGLE's GL backend**, not the WebGPU adapter handed to the page. The
recorded headless-is-SwiftShader finding stands, and a CI runner has no GPU anyway, so that lane is correctness-only.

## 48. REBASED ONTO origin/main (2026-08-22) — and main had already fixed one of our bugs

83 ahead, **0 behind**. Backup: `backup/pre-rebase-20260822` (= `eb057550`). Four conflicts, all resolved:

| file                                       | resolution                                                        |
| ------------------------------------------ | ----------------------------------------------------------------- |
| `core/test/CMakeLists.txt`                 | kept main's TSAN guard, re-added our `qa_DeviceLog` outside it    |
| `core/include/gnuradio-4.0/Graph.hpp` (×2) | took main's `emplaceBlock` signature and its `disconnectAllEdges` |
| `CMakeLists.txt`                           | kept main's relaxed plugin guard + our `GR4_REQUIRE_DEVICE` block |

### 48.1 Main independently fixed the sub-graph teardown — better than we did

`origin/main`'s `disconnectAllEdges` is **also edge-scoped now**, and additionally skips edges that are not
`Connected`, which our version did not. **Our fix is therefore superseded and was dropped**; commit
`69e11693` now carries only its benchmark. **The behaviour it existed to prove is still proven**:
`bm_fft_subgraph` reports **dataSets=1024** on all three runs against main's implementation (sub-graph -7 % vs flat,
i.e. still below the noise floor). That was the acid test of the whole rebase.

### 48.2 What main's changes forced us to adapt (`6387faed`)

`SchedulerModel` grew while we were away: `requestWorkQuiescence` → **`requestWorkQuiescenceAll`** (scheduler _and
descendants_), plus new **`blockUntilWorking()`** and **`removeBlocks()`**. For a `Domain`: quiescence stays a flag
with nothing to recurse into (its members are ordinary blocks, not schedulers), `blockUntilWorking()` is a no-op
because `startDispatch()` returns with members already RUNNING, and `removeBlocks()` drops the chain partitioning
and bindings so the next start recomputes them.

### 48.3 Two traps this rebase set

- **A dropped access specifier made `Graph::connect` private**, breaking the YAML importer. It surfaced only because
  I noticed the benchmark had run from a **stale binary** — the build had actually failed and the first "acid test
  pass" was meaningless. Check that a binary is newer than the source before believing it.
- **`qa_Graph` fails in `build-gcc15-debug` and that is NOT a regression**: main's new `groupBlocks` tests need a
  scheduler typename resolved from the block registry, and that dir has `GR_ENABLE_BLOCK_REGISTRY=OFF`. With the
  registry ON it passes (70/5). Do not chase this.

**Verified after rebase:** qa_DeviceDomain 84/11 (gcc15, clang20, acpp) · qa_DispatchGroup 24/2 ·
qa_FFTDomain 91/5 + 29/3 with the GLSL elision intact · qa_Scheduler 362/34 · qa_ManagedSubGraph 144/8 ·
qa_Graph 70/5 (registry ON) · `bm_fft_subgraph` dataSets=1024.

### 48.4 Newly available from main, relevant to this work

- **`Graph::groupBlocks` / `ungroupBlocks`** (`c94e90a3`, `3b491dbd`) — moves blocks into a subgraph with edge
  re-pointing, i.e. most of what §38.4 said auto-formation would have to build. It takes a _scheduler typename_, so
  it produces a threaded group rather than a synchronous `Domain`; the block-moving half is the reusable part.
- `ea50a53b` **fixes sample-rate forwarding for resampling blocks** — closes our long-standing `sample_rate` audit.
- `c5b75915` **the CI image now ships Firefox ESR and Chrome for Testing**, exporting `CHROME_BIN`/`FIREFOX_BIN`;
  our new `webgpu-browser` lane is their first consumer. With `emrun --browser firefox` the "no Firefox path"
  limitation is now an unused option rather than a missing capability.

## 49. RE-EVALUATION under the maintainer's four questions (2026-08-22)

Prompted by four questions after the functional review (published artifact
`claude.ai/code/artifact/e459da5e`, local copy `featDeviceIntegration_review.md`). Two of the review's
conclusions change. **Nothing here has been implemented — this is evidence for a vehicle decision that is
the maintainer's to make.**

### 49.1 `shaderFragment()` — NOT removable, and it is mis-framed in the docs

**Question:** could `shaderFragment()` be expressed with `processOne` / `processBulk` instead?

**Answer: no, and the reason is structural rather than an implementation gap.** Verified at source, not
inferred from comments:

- `ParallelFor.hpp:21-30` — `parallelFor` has a real device path for **SYCL only**. Every other backend
  falls through to a host `for` loop.
- Both framework-managed tiers (`dispatchAutoParallel`, `dispatchDeviceBulk`) begin with
  `deviceMirror(block, ctx)`, which needs `Residency::shared` — host-writable device memory holding the
  block object, because _the block is the functor_.
- `DeviceContextGLSL.hpp:82-95` and `DeviceContextWebGpu.hpp:105-115` **refuse** `Residency::shared` and
  `Residency::devicePtr`, returning an invalid `DeviceBuffer`. GL 4.3 SSBOs are opaque names, not pointers;
  a browser sandbox has no host-visible USM. Neither is a TODO — both are properties of the API.
- So on GLSL/WebGPU the framework tiers CPU-fall-back by construction, and there is no C++ → GLSL/WGSL
  compiler in the toolchain to close the other half.

**Consequence for the docs, which is the actionable part.** `USER_API_GPU_Blocks.md` lists
`shaderFragment()` as "Style 2", between the zero-effort `processOne` path and the expert
`processBulk_sycl` hatch. That ordering is wrong and misleads:

|                  | SYCL                         | GLSL / WebGPU                  |
| ---------------- | ---------------------------- | ------------------------------ |
| zero-effort path | `processOne` (auto-parallel) | **`shaderFragment()`**         |
| expert hatch     | `processBulk_sycl`           | `processBulk_glsl` / `_webgpu` |

`shaderFragment()` is the shader backends' `processOne`, not their expert hatch. The maintainer's framing —
"`_sycl`/`_cuda` are optional, expert-only" — is exactly right for the _pointer_ backends and does not
transfer to the shader ones.

**How close is it already?** Closer than the API suggests. `ShaderFragment.hpp:42-62` shows the framework
already writes every line of boilerplate — bindings, workgroup, bounds check, the dispatch loop. The block
supplies only `float process(float x) { return x * GAIN; }`, which is `processOne`'s _body_ in GLSL syntax.
The gap is the language, not the structure.

**The only real route to deleting it** is offline C++ → SPIR-V → GLSL/WGSL cross-compilation (AdaptiveCpp
already does the first leg; SPIRV-Cross does the second). That is a build-system project and a new external
dependency, which the branch constraints currently forbid. Worth recording as the long-term path; not this
PR, not the next one.

### 49.2 The managed sub-Graph vehicle — D1's "fundamental" verdict is DEAD

**This is the finding that matters most.** §D1 concluded the managed sub-Graph could not carry device
topologies because stream data never crossed the exported-port boundary, and marked it _fundamental_. The
2026-08-17 correction already argued the mechanism was actually `SchedulerBase::start()` →
`disconnectAllEdges()` running as a **port sweep** that wiped the parent's bindings.

**Main fixed it.** `Graph.hpp:843-867` is now edge-scoped with a state check. And main shipped the test:
`qa_ManagedSubGraph.cpp:257-305`, _"run a nested managed subgraph which has exported ports already connected
before being added to the scheduler"_, is the D1 RED probe almost line for line — `SlowSource →
SchedulerWrapper<Scheduler>[Copy] → CountingSink`, build-time `exportPort` + `connect`, asserting
`sink.count > firstCount`.

**Verified GREEN today: `qa_ManagedSubGraph` passes 2.35 s (clang20).**

So the maintainer's instinct — _a Graph with its own scheduler is the pattern that matches non-CPU
topologies_ — is no longer blocked by the thing that blocked it. Two honest qualifications:

1. **`Domain` already IS that pattern.** `DomainWrapper : GraphWrapper<Domain, gr::Graph>, SchedulerModel`
   — it implements the full `SchedulerModel` interface. It is a sub-graph with its own scheduler; it simply
   isn't _named_ one and doesn't route through `SchedulerWrapper`.
2. **The green test does NOT reach what the elision needs.** It asserts monotonic progress
   (`sink.count > firstCount`), with a _threaded_ scheduler inside the wrapper. The interior elision needs
   one-pass, same-quantum, single-epoch traversal (§49.3). **The boundary bridge is fixed; the scheduling
   discipline is still unproven under any `SchedulerBase`-derived driver.** Do not treat the green test as
   clearance for a vehicle pivot.

### 49.3 The real idea in the question: let the domain scheduler pass device buffers directly

> _"The sub-Graph scheduler could perhaps invoke the processing functions with the domain specific buffers
> directly?"_

This is the strongest architectural idea raised so far, and it is **not** what the code does today.

**Today, per member, per `work()` call:** `Domain::work()` → `member->work(n)` → `Block::workInternal` →
`ExecutionStrategy::dispatch` → resolve `compute_domain` string → registry lookup → pick a tier → decide
residency → maybe transfer. Every member re-discovers device-ness independently; `Domain` then _retrofits_
the shared buffer underneath via `bindDomainBuffer` + epoch stamping.

**Proposed:** the domain knows it is a device domain. It resolves residency **once**, for the whole domain,
and calls members with device buffers already in hand. What that buys:

- one residency decision per domain, not per block per call
- the transfer boundary _is_ the domain boundary, explicitly, instead of being inferred and then elided
- the epoch/stamp validity machinery (`beginDomainEpoch`, `stampDomainChain`, `domainChainHolds`) becomes
  unnecessary — it exists purely to detect that the retrofit's premise was violated
- **shader fusion across a chain becomes the scheduler's natural job** rather than an orphan

**REVERSAL — review §2.4 said delete `ShaderFusion.hpp` (156 lines).** That recommendation holds only under
the current per-block dispatch, where it is genuinely orphaned. Under the domain-scheduler direction it is
the sketch of the fusion pass. **Withdrawn if that direction is taken; stands if it is not.** The
`GLSL2WGSL.hpp` deletion (115 lines) is unaffected either way — WebGPU hand-authors WGSL.

**Cost, stated honestly:** this is a rewrite of the dispatch seam, not a refactor. It would touch
`ExecutionStrategy`, `Domain`, and every `DeviceContext`. It should not ride this PR.

### 49.5 DeviceLog first — agreed, and it splits into TWO commits

Review §2.4 said "split to its own PR". **First is better**, and the coupling analysis makes it cleaner
than expected:

| file                      | includes                                                                                                                             | can land                           |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------- |
| `DeviceLog.hpp`           | `<algorithm> <array> <cstddef> <cstdint> <format> <span> <string> <string_view> <type_traits> <utility>` — **standard library only** | **before the entire device layer** |
| `DeviceLoggerBackend.hpp` | + `Logger.hpp`, `ValueMap.hpp`, `device/DeviceContext.hpp`                                                                           | after the device runtime (PR-A)    |

So it is two commits, not one, and the first has **zero** device coupling. Landing it first means the
device work that follows can use it for diagnostics, and a small self-contained opener builds reviewer
confidence before the dispatch seam arrives.

### 49.6 What changes in the review, and what does not

| review item                          | status after this re-evaluation                                                                      |
| ------------------------------------ | ---------------------------------------------------------------------------------------------------- |
| §2.4 delete `ShaderFusion.hpp`       | **WITHDRAWN** if the domain-scheduler direction is taken; stands otherwise                           |
| §2.4 delete `GLSL2WGSL.hpp`          | unchanged — delete                                                                                   |
| §2.4 split `DeviceLog` to its own PR | **STRENGTHENED** — two commits, and first in the series                                              |
| §3.2 document the Domain (blocking)  | unchanged, and now also needs the §49.1 style-table correction                                       |
| §1 `Domain` is irreducible           | unchanged — but the _reason_ is the scheduling discipline, not the boundary bridge, which main fixed |

**Open, and now the highest-value question on the branch:** whether to keep the retrofit
(`bindDomainBuffer` + epoch stamping under an otherwise-generic dispatch) or move to a domain scheduler that
hands members device buffers directly. The first is shipped and measured; the second is cleaner, subsumes
fusion, and is a dispatch-seam rewrite. **Maintainer decision — not taken here.**

## 50. FUNCTIONAL ENVELOPE (2026-08-24) — maintainer decisions + backend re-evaluation

Supersedes §49's open questions. **Four decisions taken; the vehicle question is settled.** Nothing
implemented yet — this section scopes the work and marks every symbol keep / delete / rename.

### 50.0 Decisions taken (maintainer, 2026-08-24)

| #   | decision                                                                                                                                                                                                                        |
| --- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| D1  | **The SubGraph owns dispatch entirely.** Its internal scheduler resolves residency once and calls members' processing functions with device buffers directly. Members do NOT go through per-block `ExecutionStrategy` dispatch. |
| D2  | **Rename `Domain` → `SubGraph` throughout** — "GPU and CPU, with different scheduling domains/requirements, are the same thing".                                                                                                |
| D3  | **"No auto detection" applies to VEHICLE SELECTION only.** Chain discovery, per-edge residency probing and per-block tier selection may stay automatic.                                                                         |
| D4  | **Keep the GPU→CPU fallback, make it loud.** Nothing becomes unrunnable.                                                                                                                                                        |

**D1 and D2 interact:** D1 deletes most of the symbols D2 would rename (§50.7). Write the envelope first,
sweep once.

### 50.1 The backend question, settled at the dialect level

The maintainer asked which shader backend is "most similar to the SYCL integration". **None of them is, and
the reason is structural rather than an engineering gap.** The deciding axis is not the API — it is what
artefact the backend can consume, and in which SPIR-V dialect.

| backend                  | consumes                                                                                                     | dialect                                    | can C++ ever be the kernel?                    |
| ------------------------ | ------------------------------------------------------------------------------------------------------------ | ------------------------------------------ | ---------------------------------------------- |
| **SYCL / AdaptiveCpp**   | LLVM IR, JIT-ed at runtime                                                                                   | n/a — IR, not SPIR-V                       | **yes** — C++ _is_ the kernel                  |
| **native GL 4.6**        | GLSL source, **or** SPIR-V (`glShaderBinary` + `glSpecializeShader`, core since 4.6; `ARB_gl_spirv` on 4.3+) | **Shader** (GLCompute, Logical addressing) | no — wrong dialect                             |
| **native WebGPU (Dawn)** | WGSL; SPIR-V only as a transitional measure being removed                                                    | Shader                                     | no — and do not build on the transitional path |
| **browser WebGPU**       | **WGSL source only**, permanently                                                                            | n/a                                        | **no**                                         |

Two independently verified facts do the work:

1. **SPIR-V for shaders and SPIR-V for kernels are disjoint subsets with no interoperability.** OpenCL uses
   the `Kernel` execution model with `Physical32/64` addressing; Vulkan/GL compute uses `GLCompute` with the
   `Shader` capability and `Logical` addressing. Vulkan will not run Kernels; OpenCL will not run Shaders,
   "not even compute ones".
2. **AdaptiveCpp SSCP does not emit SPIR-V at build time at all.** Observed in
   `build-acpp/.../qa_HostToDevice.cpp.o`: the embedded `__acpp_local_sscp_hcf_content` blob declares
   `generator=hipSYCL SSCP`, `format=llvm-ir`. It is **LLVM IR**, JIT-compiled at runtime to whatever device
   is present (PTX for CUDA, Kernel-model SPIR-V for Level Zero/OpenCL, host code for OMP).

So the chain _C++ → SYCL → SPIR-V → GL/WebGPU_ is blocked **twice**: there is no build-time SPIR-V to hand
over, and if one were forced it would be the wrong dialect. This is not a TODO.

### 50.2 `shaderFragment()` — re-evaluated carefully; the maintainer's reading is CORRECT

**Verdict: `shaderFragment()` is a strong requirement for GPU-accelerated processing on native-GL and in the
browser, not an optimisation hatch.** The mechanism, verified at source:

- `ParallelFor.hpp:21-30` — a real device path exists for **SYCL only**; every other backend falls through
  to a host `for` loop.
- Both framework tiers (`dispatchAutoParallel`, `dispatchDeviceBulk`) open with `deviceMirror(block, ctx)`,
  needing `Residency::shared` — host-writable device memory holding the block object, _because the block is
  the functor_.
- `DeviceContextGLSL.hpp:82-95` and `DeviceContextWebGpu.hpp:105-115` **refuse** `Residency::shared` and
  `Residency::devicePtr`, returning an invalid `DeviceBuffer`. GL 4.3 SSBOs are opaque names, not pointers;
  a browser sandbox has no host-visible USM. Neither is a TODO — both are API properties.
- Even granting a device pointer, §50.1 shows the kernel body cannot be C++.

**Nuance worth recording:** requesting **GL 4.6** instead of the current 4.3 (`GlComputeContext.hpp:138`
asks for `EGL_CONTEXT_MAJOR_VERSION 4, MINOR 3`) would let the GL backend ingest SPIR-V. That does _not_
rescue `processOne` — the dialect is still wrong — but it would let a **build-time-generated** kernel be
shipped as a binary rather than as source text. Note only; not a recommendation for this PR.

**The framing must change in the docs.** `USER_API_GPU_Blocks.md` lists `shaderFragment()` as "Style 2",
between the zero-effort path and the expert hatch. Correct table:

|                  | SYCL (native)                      | GLSL / WebGPU                  |
| ---------------- | ---------------------------------- | ------------------------------ |
| zero-effort path | `processOne`, `processBulk`        | **`shaderFragment()`**         |
| expert hatch     | `processBulk_sycl` (later `_cuda`) | `processBulk_glsl` / `_webgpu` |

The maintainer's framing — "`_sycl`/`_cuda` are optional, expert-level" — is exactly right for the pointer
backends and does not transfer to the shader ones.

**How close is it already?** `ShaderFragment.hpp:42-62`: the framework writes every line of boilerplate —
bindings, workgroup size, bounds check, dispatch loop. The block supplies only
`float process(float x) { return x * GAIN; }` — `processOne`'s _body_ in GLSL syntax. **The gap is the
language, not the structure.** That is the smallest honest statement of the cost.

### 50.3 WebGPU evaluated independently — it is GLSL-like, not SYCL-like

Asked separately because the newer API might have closed the gap. **It has not, and in the browser it is
closed permanently.**

- **Residency:** identical to GL. `DeviceContextWebGpu.hpp:105-115` honours only `Residency::host` (heap)
  and `Residency::opaque` (a `GPUBuffer` handle). No host-visible USM in a browser sandbox — a sandbox
  property, not a Dawn limitation.
- **Shader ingestion:** browsers accept **WGSL source only**. SPIR-V is not exposed to web content; the
  gpuweb SPIR-V execution-environment spec has _itself transitioned_ from "directly ingestible" to "should
  be translatable into WGSL". Some native implementations accept SPIR-V transitionally and will remove it.
- **Consequence:** WebGPU sits in the same column as GLSL in §50.1. It is a better _engineered_ backend than
  GL compute (explicit pipelines, real queues, no EGL/driver roulette, and it is the only GPU compute path
  in a browser at all — WebGL2 is GLES 3.0 and has **no compute shaders**), but it is not architecturally
  closer to SYCL.

**Recommendation:** treat WebGPU as the strategic shader backend and GL compute as the native fallback where
no SYCL toolchain exists. They share a residency model, a dispatch model and a fragment-source requirement,
so one shader-side abstraction serves both — which is what the SubGraph scheduler should target (§50.6).

### 50.4 What would it take to run C/C++ in a browser, on the GPU?

Decomposed into four questions, because the answers differ:

| question                  | answer                                                                                                                                                                                                                                                |
| ------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| what **runs** the kernel? | WebGPU. WASM has no GPU access of its own; WebGL2 has no compute shaders. There is no third option.                                                                                                                                                   |
| what **compiles** it?     | Nothing available compiles C++ to WGSL. The dialect wall (§50.1) rules out the clang→SPIR-V→Tint route, because clang emits Kernel-model.                                                                                                             |
| **when**?                 | Necessarily build time — a browser cannot host a C++ compiler for this.                                                                                                                                                                               |
| what **owns the buffer**? | Correct, `CircularBuffer<T>` is not required — `Residency::opaque` `GPUBuffer` handles already work, and `InputViewLike`/`OutputViewLike` (`Buffer.hpp:73-79`) already name a kernel-facing view with no accounting. **This part is already solved.** |

**So the honest answer: the kernel body must exist as WGSL text at some point; the only open question is who
writes it and when.** Three routes:

1. **The user writes it** — `shaderFragment()`. Shipped, working, measured in-browser on hardware.
2. **A build-time tool generates it from a restricted C++ subset** — a real project (parse a constrained
   `processOne`, emit WGSL). Buys "write it once" for element-wise blocks; does not extend to blocks with
   shared memory, barriers or multi-stage kernels.
3. **A small expression DSL** emitting both C++ and WGSL. Rejected — §8.1/§8.2 of the style guide, and it
   would constrain every block author to serve a minority backend.

Route 2 is the only one that could ever retire `shaderFragment()`, and it is a separate project with its own
PR. **Recommendation: keep route 1 as the contract; record route 2 as the long-term path.**

### 50.5 The GPU→CPU fallback — decision D4, and what "loud" has to mean

**CORRECTED 2026-08-24 — an earlier draft of this section had the maintainer's concern backwards.** The
nuisance being guarded against is **hard failure**: a graph that errors and stops on a GPU-less machine.
_"In many cases the speed difference is only marginal and having at least some numerics output (even if
slow) is better than none."_ Slow-but-running is the DESIRED outcome, not the problem. Do not write a
fail-fast path, and do not describe the fallback as a defect.

**Pros of keeping it (the decisive ones):** a GPU-authored graph stays runnable on a laptop, in CI and in a
demo; degraded numerics beat absent numerics; the device tests run on GPU-less runners; nothing becomes
unrunnable by accident. **The only genuine con:** the user may not realise they are on the slow path — so
loudness is about _informing_, never about _stopping_.

**So "loud" has to do more than today's log line.** Options, cheapest first — a decision is still needed:

| mechanism                                                                  | what it buys                                                                                               | cost                                                    |
| -------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- | ------------------------------------------------------- |
| **graph-level summary at `start()`**                                       | one message naming every block that wanted a device and did not get one, before any samples move           | ~20 lines; needs a collection point                     |
| **queryable state** on the SubGraph / scheduler (e.g. `degradedDomains()`) | a UI or a test can assert it; not missable programmatically                                                | small; new public surface                               |
| **a tag on the first output sample**                                       | travels with the data to a sink or a monitor                                                               | fits the existing tag path; per-graph noise if repeated |
| **`GR4_REQUIRE_DEVICE`-style env gate**                                    | already built (`DeviceExpectation.hpp`) — turns a silent degradation into a hard failure for CI/production | zero — already shipped, currently test-only             |

**Recommendation:** graph-level summary at `start()` as the DEFAULT — informative, never fatal. The
existing `GR4_REQUIRE_DEVICE` gate may additionally be promoted from test-only to a supported knob for CI
and for users who explicitly want degradation to be fatal, but it must stay **opt-in and off by default**:
per the correction above, refusing to run is the failure mode being avoided, not a feature.

### 50.6 What D1 (SubGraph owns dispatch) actually requires

Per member, the SubGraph scheduler must: pick the member's tier from its traits and the resolved backend,
hand it buffers of the domain's residency, and sequence it. What it must **not** do is let the member
re-resolve `compute_domain` and re-decide residency per call.

| backend       | what the SubGraph scheduler hands a member                                                                                               |
| ------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| SYCL          | USM pointers; members are kernels submitted to one queue; interior edges never leave the device                                          |
| GLSL / WebGPU | SSBO / `GPUBuffer` handles; members contribute `shaderFragment()`s that the scheduler **fuses** into one dispatch where the chain allows |
| host          | ordinary spans — identical to today                                                                                                      |

**Shader fusion becomes the scheduler's job**, which is the natural home for it and the reason the §2.4
review recommendation to delete `ShaderFusion.hpp` (156 lines) is **WITHDRAWN** — see §49.3. `GLSL2WGSL.hpp`
(115 lines) deletion is unaffected.

`ExecutionStrategy` is **not** deleted: its trait machinery (`HasSyclBulkForSpans`, `AutoParallelisable`,
`DeviceRelocatable`, `firstNonRelocatableMember`) is what tells the SubGraph scheduler which tier a member
offers. What splits off is the per-block residency/transfer body, which the per-block vehicle keeps and the
SubGraph vehicle replaces.

### 50.7 Symbol disposition — keep / delete / rename

Written before the sweep so it happens once.

| symbol                                                                                                                                                            | occurrences | disposition                                                                                                                                                                                                                                                                                                                                                     |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `gr::device::Domain`                                                                                                                                              | 2           | **rename** → `gr::SubGraph`                                                                                                                                                                                                                                                                                                                                     |
| `DomainWrapper<T>`                                                                                                                                                | 6           | **rename** → `SubGraphWrapper<T>`                                                                                                                                                                                                                                                                                                                               |
| `makeDomain()`                                                                                                                                                    | 32          | **rename** → `makeSubGraph()`                                                                                                                                                                                                                                                                                                                                   |
| `DomainHandle`                                                                                                                                                    | 4           | **rename** → `SubGraphHandle`                                                                                                                                                                                                                                                                                                                                   |
| `Domain::work()`                                                                                                                                                  | —           | **keep, rewrite** — drives members' processing functions, not `member->work()`                                                                                                                                                                                                                                                                                  |
| `linearChains()`, `topologicalOrder()`                                                                                                                            | —           | **keep** — D3 leaves chain discovery automatic                                                                                                                                                                                                                                                                                                                  |
| `startDispatch`/`stopDispatch`/`removeMembers`/quiescence                                                                                                         | —           | **keep**                                                                                                                                                                                                                                                                                                                                                        |
| `debug_fill_host_rings`                                                                                                                                           | —           | **keep** — more useful once the SubGraph owns the buffers                                                                                                                                                                                                                                                                                                       |
| `bindChainBuffers()`                                                                                                                                              | —           | **DELETE** — the retrofit; the SubGraph owns buffers directly                                                                                                                                                                                                                                                                                                   |
| `DomainBinding` + `bindDomainBuffer` / `clearDomainBindings` / `beginDomainEpoch` / `stampDomainChain` / `domainChainHolds` / `nextChainId` / `pooledChainBuffer` | 21 + 11     | **DELETE** — all exist only to make the retrofit safe. D1 removes the premise they guard.                                                                                                                                                                                                                                                                       |
| `compute_domain`                                                                                                                                                  | many        | **KEEP UNCHANGED — recommended, see §50.11.** Public/SigMF-facing selector that does two jobs (thread pool + device); renaming breaks saved graphs and YAML for no semantic gain.                                                                                                                                                                               |
| files `device/DeviceDomain.hpp`, `qa_DeviceDomain.cpp`, `qa_FFTDomain.cpp`                                                                                        | —           | rename to `SubGraph.hpp`, `qa_SubGraph.cpp`, `qa_FFTSubGraph.cpp`                                                                                                                                                                                                                                                                                               |
| namespace / location                                                                                                                                              | —           | **DECIDED (maintainer, 2026-08-24): `gr::SubGraph` in `core/include/gnuradio-4.0/SubGraph.hpp`.** Under D1 it is the general vehicle and drives host members identically to device ones. Consequence for the PR: this is a NEW CORE header, not an additive device one, so the review's "9 modified core headers" figure gains a 10th file (new, not modified). |

**Net:** the D1 rewrite _deletes_ ~32 call sites of epoch/stamp machinery. The prune and the architecture
pull in the same direction.

### 50.8 Functional envelope

| capability                                                                  | status                                                                                                                 |
| --------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| `processOne` / `processBulk` unchanged on CPU                               | **IN** — shipped                                                                                                       |
| `processBulk(InputViewLike auto…, OutputViewLike auto…)` no-accounting form | **IN** — concepts already exist (`Buffer.hpp:73-79`); needs wiring + docs, not design                                  |
| `processOne` auto-parallel on SYCL                                          | **IN** — shipped, gated by `static_assert`                                                                             |
| `shaderFragment()` for GLSL/WebGPU                                          | **IN** — and reclassified as the zero-effort path for those backends, not a hatch                                      |
| `processBulk_sycl` expert hatch                                             | **IN** — expert-level, explicitly optional                                                                             |
| `processBulk_cuda`                                                          | **OUT** — future extension point, no code                                                                              |
| per-block device dispatch (USM-PMR `CircularBuffer`)                        | **IN** — one of the two vehicles                                                                                       |
| SubGraph + internal scheduler owning dispatch                               | **IN** — D1; the rewrite                                                                                               |
| automatic vehicle selection                                                 | **OUT** — D3                                                                                                           |
| automatic chain discovery inside a SubGraph                                 | **IN** — D3 permits it                                                                                                 |
| GPU→CPU fallback                                                            | **IN** — D4, with §50.5 loudness                                                                                       |
| C++ → WGSL/GLSL kernel generation                                           | **OUT** — §50.1 dialect wall; route 2 of §50.4 is a separate project                                                   |
| `DeviceLog`                                                                 | **IN, FIRST** — two commits; `DeviceLog.hpp` is standard-library-only and lands ahead of the device layer              |
| `GLSL2WGSL.hpp` (115)                                                       | **OUT** — delete                                                                                                       |
| `ShaderFusion.hpp` (156)                                                    | **IN** — deletion withdrawn; it is the fusion pass D1 needs                                                            |
| sub-graph auto-formation                                                    | **DROPPED FROM SCOPE (maintainer, 2026-08-25)** — a helper, needs no core change, later PR if wanted (§38)             |
| tags in auto-parallel kernels                                               | **DEFERRED** — blocked on the `kBlobAlignment` bug                                                                     |
| `processEpilogue` on the device path                                        | **DROPPED FROM SCOPE (maintainer, 2026-08-25)** — a CPU concern (SIMD batch sizes, tag-split frames); not a device one |

### 50.10 Sequence

> **ALL SEVEN DONE.** 1-2 §58.10 (rename folded so `Domain` enters no commit) · 3 docs landed with commit 10 ·
> 4 D1 landed as the `SubGraph` owning its own dispatch, and the epoch/stamp machinery went with the chain
> machinery (§58.4) · 5 §50.5 — `ExecutionStrategy.hpp:167-171` names the block and the reason · 6 `GLSL2WGSL.hpp`
> gone with D5, `DeviceLog` landed as commit 7 plus the `LogRecord` refactor in commit 14 · 7 §58.

1. confirm the two open naming points (§50.7: `compute_domain`, and `gr::` vs `gr::device::`)
2. `Domain` → `SubGraph` sweep — once, using the §50.7 dispositions
3. docs: the §50.2 style table + the SubGraph section the review flagged as blocking
4. D1 rewrite: SubGraph scheduler owns dispatch; delete the epoch/stamp machinery
5. fallback loudness (§50.5)
6. prune: `GLSL2WGSL.hpp`; `DeviceLog` split to two commits at the front
7. history refactor to the thematic cut, now that the envelope fixes what each PR contains

### 50.11 `compute_domain` — exact definition, and the overloading it hides

Asked by the maintainer before settling the rename. **Recommendation: keep the name.**

**Declaration** (`Block.hpp:742`), whose own Doc string admits the dual role:

```cpp
A<std::string, "compute domain", Doc<"compute domain/IO thread pool name">>
    compute_domain = gr::thread_pool::kDefaultIoPoolId;   // default = "default_io", NOT "host"
```

Reflected at `Block.hpp:784`, so it is public, YAML/SigMF-facing and runtime-settable.

**Two jobs in one string:**

1. **CPU thread-pool affinity** — `default_io`, `default_cpu`, or a user pool name.
2. **Device selection** — grammar `kind[:backend[:deviceIndex]]`, kinds `host|gpu|fpga|tpu`, e.g. `gpu:sycl:0`.

**Disambiguation** is `ComputeDomain::parse()` (`ComputeDomain.hpp`): `""`/`host`/`default_cpu`/`default_io`
map to `host()`; an **unrecognised kind maps to `host()` silently**, falling through to "it must be a pool
name". `isDevice()` is `kind != "host" || backend != "none"` — note `host:sycl` IS a device (host-resident
memory, SYCL execution).

**Consumers (four):**

| site                                             | use                                                                                                          |
| ------------------------------------------------ | ------------------------------------------------------------------------------------------------------------ |
| `Graph.hpp:721-733`                              | auto-populates the **edge** domain → selects the edge buffer's PMR (device > block > edge > graph > default) |
| `Block::migrateFieldsToDeviceResource()` `:1015` | re-seats the block's own pmr fields onto the device resource                                                 |
| `ExecutionStrategy::dispatch`                    | resolves via `DeviceContextRegistry` to pick the backend                                                     |
| `makeDomain` / `bindChainBuffers`                | chain-domain agreement; the two-device-domain refusal                                                        |

**Why keep the name:** it describes both jobs accurately, it is the public SigMF selector named in goal 4,
and a rename breaks every saved graph and YAML flowgraph for no semantic gain.

**But flag the overloading — it is a live D4 problem.** The warn-once for a string that looks like a domain
but does not parse fires **only when the string contains a colon** (`Block.hpp:990`). So `gpu:sycl` mistyped
as `gpu_sycl` silently becomes a host thread pool _named_ `gpu_sycl`: no warning, no device, just slow. That
is the same silent-demotion failure class D4 asked to be made loud, and it is worse than the fallback itself
because **nothing logs at all**. Fold into the §50.5 loudness work: a `compute_domain` that names neither a
registered pool nor a parseable domain should say so at `start()`, colon or no colon.

**Separate, larger question, not for this PR:** whether one string should carry both "which CPU pool" and
"which GPU" at all. D3 makes the _vehicle_ explicit while this string still demotes silently — worth its own
look once the envelope is executed.

## 51. THE ExprTk QUESTION — a limited language that runs on the GPU (2026-08-24)

**Maintainer's question, and it is a different question from §50.1.** Not "map C++ through SYCL/SPIR-V", but:
_is there an ExprTk-like strategy — C-like ASCII parsed to an AST — that can execute on WebGPU?_

**This sidesteps the §50.1 dialect wall entirely, because it never produces SPIR-V.** It produces WGSL/GLSL
**text**, which is exactly what those backends want. §50.1 remains true and is simply not the relevant
constraint here. My earlier framing was too narrow.

**Answer: yes — and one option is a Khronos project rather than something we would build.**

### 51.1 The option space, most mature first

| approach                                      | user writes                          | targets                                            | maturity                          | new dependency                           |
| --------------------------------------------- | ------------------------------------ | -------------------------------------------------- | --------------------------------- | ---------------------------------------- |
| **today** (`shaderFragment`)                  | GLSL fragment, **plus WGSL by hand** | GL, WebGPU                                         | shipped, measured                 | none                                     |
| **GLSL → WGSL transpile**                     | GLSL fragment only                   | GL, WebGPU                                         | mature libraries                  | +1 (ShaderTranspiler / Naga / Tint)      |
| **Slang**                                     | one Slang function                   | GL, WebGPU, SPIR-V, HLSL, Metal, CUDA, **and CPU** | Khronos-hosted, production-proven | +1 (Slang compiler; a WASM build exists) |
| bespoke expression DSL (true ExprTk analogue) | C-like expression string             | whatever we emit                                   | **does not exist — we build it**  | none                                     |
| C++ expression templates (Halide-style)       | real C++ operators                   | whatever we emit                                   | **does not exist — we build it**  | none                                     |

### 51.2 Slang is the strongest answer, and it hits the primary goal directly

[Slang](https://shader-slang.org/) is an open-source compiler [hosted by Khronos, contributed by
NVIDIA](https://www.khronos.org/news/press/khronos-group-launches-slang-initiative-hosting-open-source-compiler-contributed-by-nvidia).
From **one** C-like source it generates SPIR-V, HLSL, GLSL, **WGSL**, Metal, CUDA — _and code that runs on a
CPU_.

**That last item is why this matters more than a transpiler.** The branch's primary goal is that writing a
processing function be near-identical for CPU and GPU. Slang reaches that goal by moving the kernel body to
**one language that already targets both**, instead of by translating C++ into something a GPU accepts. It
is a different route to the same objective, and it does not hit the dialect wall.

Maturity evidence, not marketing: Valve compiled the **entire production Source 2 HLSL codebase** with Slang
while modifying 10 lines. And [a WASM build exists](https://github.com/semisgdh/slang-playground-wasm), so
the compiler can run **in the browser** — meaning kernel generation could be runtime, not only build-time,
on the WASM target.

**The honest cost:** the user writes _Slang_, not C++. It is C-like and HLSL-derived, so it is far closer to
`processOne` than GLSL is, but it is still a second language in the codebase. And it is a new external
dependency, which the §2 constraints currently forbid ("no external GPU library dependencies beyond
AdaptiveCpp") — a constraint written before this option was on the table, and worth revisiting on its
merits rather than by default.

### 51.3 The cheap intermediate: stop hand-authoring WGSL

Independently of Slang, **GLSL → WGSL is a solved, packaged problem**:
[ShaderTranspiler](https://github.com/RavEngine/ShaderTranspiler) (C++, GLSL → HLSL/Metal/Vulkan/WebGPU),
Naga (a translation layer across WGSL / SPIR-V / GLSL / MSL / HLSL), and Tint.

**This reframes `GLSL2WGSL.hpp`.** §50.8 lists it as OUT — 115 lines, test-only, a hand-rolled toy. That
deletion still stands, but the _function it was reaching for is real and worth having_: with a proper
transpiler, a block writes **one** GLSL fragment and both GL and WebGPU are served. Today `fft.hpp` carries
a GLSL path and a separately hand-authored WGSL path.

**So the choice is not "delete or keep our toy" — it is "delete the toy and adopt a real transpiler, or
delete the toy and keep hand-authoring both".** That is a maintainer decision the envelope did not previously
put on the table.

### 51.4 The true ExprTk analogue does not exist off the shelf

Searched for a library doing runtime expression parsing → GPU kernel emission. **Nothing found.** The pieces
exist separately — GLSL parsers producing ASTs (`glsl-parser`, C++03, ~90 KB, MIT), Naga for translation —
but no packaged "expression string in, WGSL out".

Building one is feasible for **element-wise** blocks, and closer than it looks: `ShaderFragment.hpp:42-62`
already generates the entire shader around a single `float process(float x)` body. An ExprTk-style front end
would replace only that one function body — parse `x * gain + offset`, emit the GLSL/WGSL for it.

**But it does not extend past element-wise.** Chunked and multi-stage kernels — the FFT is the branch's own
example, with its Stockham stages, shared memory and barriers — are not expressions, and no expression
language will describe them. So this route would serve the simple blocks and leave the interesting ones
exactly where they are.

### 51.5 Recommendation

1. **Delete `GLSL2WGSL.hpp`** — unchanged from §50.8.
2. **Put GLSL → WGSL transpilation on the table as a decision** (§51.3). Cheapest real win: halves the
   shader-authoring burden, removes the hand-written WGSL in `fft.hpp`, costs one build-time dependency.
3. **Evaluate Slang properly as a follow-up spike**, not in this PR. It is the only option that reaches the
   primary goal — one kernel source for CPU and every GPU backend — and being Khronos-hosted it is unlikely
   to evaporate. The spike should answer: build-time or runtime on WASM; how a Slang kernel binds to
   `InputViewLike`/`OutputViewLike`; whether its CPU target is fast enough to be the CPU path or only a
   reference; and what it does to build times.
4. **Do not build a bespoke expression DSL** (§51.4). It cannot express the blocks that motivated the work,
   and Slang already covers the ground it would cover.
5. `shaderFragment()` stays the contract meanwhile (§50.2) — every route above changes _what language goes
   inside it_, not the seam itself. That is a useful property: none of these decisions has to be taken now.

**Constraint to revisit:** §2's "no external GPU library dependencies beyond AdaptiveCpp" predates both
WebGPU and Slang. Items 2 and 3 each need exactly one build-time dependency. Worth an explicit decision
rather than an inherited default.

### 51.6 ShaderTranspiler does NOT take C++ — and §50.1 needs a correction

**Direct answer: no.** [ShaderTranspiler](https://github.com/RavEngine/ShaderTranspiler) accepts **GLSL only**
and emits ESSL, HLSL, DXIL, MSL, MSL-binary, SPIR-V and WGSL (it wraps glslang + SPIRV-Cross). It is a C++
_library that transpiles GLSL_, not a transpiler _of_ C++. §51.3 described it as "C++, GLSL → …", which was
ambiguous — the "C++" was the implementation language.

**But the question exposed a real over-generalisation in §50.1, which is corrected here.**

§50.1 said: SPIR-V for shaders and SPIR-V for kernels are disjoint subsets, therefore C++ can never be the
kernel on GL/WebGPU. **The first half is right; the second does not follow.** The disjointness blocks
**artefact interchange** — you cannot hand Kernel-model SPIR-V (what a SYCL toolchain produces) to Vulkan or
GL. It does **not** block **compilation**: a compiler is free to target the Shader dialect from C-family
source. Three existence proofs:

| compiler                 | input                     | output                                          | evidence                                                                                                                                                                                                 |
| ------------------------ | ------------------------- | ----------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Circle** (Sean Baxter) | **real C++**              | SPIR-V incl. **compute** shaders                | lowers C++ shader functions to SPIR-V; GLSL builtins and vector/matrix types built into the compiler                                                                                                     |
| **clspv** (Google)       | OpenCL C                  | **Vulkan** SPIR-V                               | production dialect bridge, open source, maintained                                                                                                                                                       |
| **LLVM SPIR-V backend**  | LLVM IR / clang frontends | SPIR-V, with **official Vulkan target triples** | `spirv64-unknown-vulkan`, `spirv32-unknown-vulkan1.3`, `vulkan1.2`/`1.3` OS options ([LLVM docs](https://llvm.org/docs/SPIRVUsage.html)); "Vulkan compute is the most well supported use case right now" |

So the corrected §50.1 claim is: **the SYCL route to GL/WebGPU is blocked, because AdaptiveCpp emits LLVM IR
JIT-ed to Kernel-model targets and that artefact cannot be reused. A _separate_ compilation path from a
restricted C++ subset to Shader-model SPIR-V is possible, and upstream LLVM has the target.**

### 51.7 Assessed against GR4's actual constraints

| route                   | input              | chain                                                   | toolchain fit                                                     | verdict                                                                                           |
| ----------------------- | ------------------ | ------------------------------------------------------- | ----------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| ShaderTranspiler        | GLSL               | GLSL → WGSL                                             | fine, build-time                                                  | **useful, but not C++** — §51.3 stands                                                            |
| **LLVM SPIR-V backend** | **restricted C++** | clang → SPIR-V(Vulkan) → Tint/Naga → WGSL               | **we already require Clang 20**                                   | **the only literal-C++ route worth a spike**                                                      |
| Circle                  | real C++           | Circle → SPIR-V → WGSL                                  | **breaks the compiler matrix** — a separate, proprietary compiler | **not viable**: closed source, one author, outside GCC 15 / Clang 20 / Emscripten                 |
| clspv                   | OpenCL C           | clspv → SPIR-V(Vulkan) → WGSL                           | fine, build-time                                                  | viable but OpenCL C is another language, and strictly less capable than Slang                     |
| **Slang**               | Slang              | slangc → **WGSL directly** (+ CPU, SPIR-V, CUDA, Metal) | fine, build-time                                                  | **still the best value** — no SPIR-V hop, no dialect question, and the only one targeting CPU too |

**Honest caveat on the LLVM route, and it is the load-bearing one.** The Vulkan _target triples_ exist and
are documented. What is **not** established is that clang's ordinary C++ frontend will lower general C++ to
them — that path is driven primarily by clang's **HLSL** frontend, and the SPIR-V Vulkan environment demands
**Logical addressing**: no pointer arithmetic across objects, no dynamic allocation, no virtual dispatch, no
exceptions, effectively no `std::`. That is not "write ordinary `processOne`". It is precisely the _limited
C/C++ subset_ the maintainer asked about — so the ask is well matched — but the subset is severe and its
boundary is undocumented. **A spike would have to establish it empirically.**

### 51.8 The point that actually matters: two separable concerns

The dialect question and the residency question are **independent**, and conflating them is what made §50.2
read as more final than it is:

1. **What language the kernel body is written in.** Solvable — Slang today, possibly restricted C++ via LLVM.
2. **How the block's settings reach the kernel.** GR4's framework tiers pass _the block itself_ as the
   functor (`dBlock->processOne(dIn[i])`), which needs a host-writable device pointer to a mirrored block
   object. GL and WebGPU refuse that (`Residency::shared` → invalid), and **Logical addressing forbids it in
   principle**, so no compiler route changes it.

**Consequence:** even a perfect C++ → WGSL path would not give `processOne`-on-WebGPU as written today. It
would give a way to author the _kernel body_ in C++, with settings arriving as **specialisation constants or
uniforms** instead of through a mirrored object — which is exactly what `ShaderFragment::constants` already
does. So the seam does not change; the language inside it does. §50.2's conclusion that `shaderFragment()`
is the shader backends' zero-effort path **stands**, but for the residency reason, not the dialect reason.

### 51.9 Revised recommendation

1. **Slang remains the primary candidate** (§51.2) — emits WGSL directly, no SPIR-V hop, no dialect
   question, and uniquely targets CPU as well.
2. **Add one spike: restricted C++ → `spirv64-unknown-vulkan` → WGSL with Clang 20.** It is the only route
   where the kernel body stays literal C++ _and_ the toolchain is one GR4 already requires. Success
   criterion: compile a trivial element-wise kernel through clang to Vulkan SPIR-V, translate to WGSL, and
   run it in the browser against the existing WebGPU context. Failure is cheap and informative — it would
   pin the subset boundary, which nobody has documented.
3. **Circle: no.** Technically the most impressive (real C++, compute shaders), but closed source, single
   author, and it would mean compiling shader TUs with a compiler outside the supported matrix.
4. **clspv: no** — dominated by Slang on every axis that matters here.
5. **§51.5 items 1, 2, 4 and 5 unchanged.**

## 52. HOW SLANG WOULD MEET `processOne` / `processBulk` (2026-08-24)

Maintainer's question. **Short answer: Slang does not change the required interface — it replaces what fills
the shader-backend slot. It collapses GLSL + hand-written WGSL into ONE GPU body and adds SPIR-V, CUDA and
Metal for free. It does NOT unify CPU and GPU authoring, and claiming otherwise would be the mistake.**

### 52.1 The three things any bridge must solve

|                       | GR4 contract                                   | Slang model                                                   |
| --------------------- | ---------------------------------------------- | ------------------------------------------------------------- |
| where the body lives  | C++ member function                            | module-scope function / entry point                           |
| how settings reach it | direct member access (`x * gain`)              | explicitly-bound `uniform` / `ParameterBlock` / push constant |
| who owns the loop     | `processOne` is per-sample; framework wraps it | entry point is per-thread, `SV_DispatchThreadID`              |

### 52.2 Shape 1 — Slang as the shader-backend body (RECOMMENDED)

A drop-in replacement for `shaderFragment()`'s payload. **The interface is untouched.**

```cpp
struct Gain : gr::Block<Gain> {
    gr::PortIn<float>  in;
    gr::PortOut<float> out;

    Annotated<float, "gain"> gain = 1.0f;

    GR_MAKE_REFLECTABLE(Gain, in, out, gain);

    [[nodiscard]] constexpr float processOne(float x) const noexcept { return x * gain; }   // CPU, unchanged

    static constexpr std::string_view kernelSlang = R"(
        float process(float x, float gain) { return x * gain; }
    )";
};
```

The framework generates the entry point around it, exactly as `generateElementWiseShader` does today:

```slang
struct Params { float gain; };
ParameterBlock<Params>    params;
StructuredBuffer<float>   inBuf;
RWStructuredBuffer<float> outBuf;
uniform uint              count;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= count) { return; }
    outBuf[tid.x] = process(inBuf[tid.x], params.gain);
}
```

`slangc` compiles that once at build time to **WGSL + SPIR-V + GLSL**, all cached. `DeviceContextGLSL` and
`DeviceContextWebGpu` each pick their artefact; nothing else in the dispatch path changes.

**What this actually buys:**

1. **One GPU body instead of two.** `fft.hpp` today carries a GLSL path _and_ a separately hand-authored
   WGSL path. This is the concrete duplication Slang removes.
2. **Settings binding becomes automatic and type-safe.** Today `ShaderFragment::constants` is
   `std::vector<ShaderConst>` where `ShaderConst = {std::string name; float value;}` — **floats only,
   hand-listed by the block author**. Slang has a [reflection API](http://shader-slang.org/slang/user-guide/reflection)
   reporting parameter layout per target. `GR_MAKE_REFLECTABLE` already knows the block's settings, so the
   framework could bind them into a `ParameterBlock` automatically, at any type. That is a strict
   improvement over what ships today, independent of everything else.
3. **The shader path stops being element-wise-only.** `generateElementWiseShader` can express exactly one
   shape: `out[i] = process(in[i])`. Slang supports `groupshared` memory and barriers, so chunked and
   multi-stage kernels — the FFT's Stockham stages, the branch's own motivating example — become
   expressible. `ShaderFragment` already carries `inputChunkSize`/`outputChunkSize` for this and nothing
   consumes them meaningfully. **This is the biggest capability gain, and it is bigger than the
   WGSL-deduplication.**
4. Free extra targets: SPIR-V, CUDA, Metal from the same source.

**What it does NOT buy:** the maths is still written twice — once as C++ `processOne`, once as Slang. Slang
solves _GPU-side portability_, not _CPU/GPU unification_.

### 52.3 Shape 2 — Slang as the single source, CPU via Slang→C++ (REJECTED)

Slang [can emit C++ source](http://shader-slang.org/slang/cpu-target.html) and has host-callable and
shared-library CPU targets, so "write the kernel once in Slang, generate the CPU path too" is _technically_
on the table. **Reject it.** Three reasons, in order of severity:

1. **It breaks the required interface rather than working with it.** `processOne` is not merely a place to
   put maths — it is what `HasNoexceptProcessOneFunction`, the SIMD gate, `constexpr` evaluation and
   inlining into the scheduler's hot loop all key off. A generated function the C++ compiler sees only as
   emitted source loses the `processOne(simd<T>)` overload path and the constant-folding that makes the CPU
   path fast. **The CPU path is GR4's main path**; degrading it to serve a minority backend is the wrong
   trade.
2. **Slang's C++ emission is self-described as preliminary.** Not a foundation for the primary code path.
3. It puts a codegen step in front of every block, for every build, including CPU-only builds — directly
   against the "keep it MVP for users who do not use device compute" decision.

### 52.4 Shape 3 — generate the C++ `processOne` into the block (REJECTED)

Codegen into user headers so the user writes Slang and the framework synthesises `processOne`. Invasive,
fragile, and it makes every block's source a build artefact. No.

### 52.5 `processBulk` specifically

`processBulk(InputViewLike auto…, OutputViewLike auto…)` — the no-accounting form (§50.8) — maps to Slang
**better than `processOne` does**, because it is already span-shaped and the entry point is already
thread-indexed:

```cpp
work::Status processBulk(InputViewLike auto in, OutputViewLike auto out) noexcept;   // CPU

static constexpr std::string_view kernelSlang = R"(
    void processChunk(StructuredBuffer<float> in, RWStructuredBuffer<float> out, uint base, uint n) { ... }
)";
```

The framework supplies the dispatch geometry from `inputChunkSize`/`outputChunkSize`. This is the shape the
FFT needs and the one the current shader path cannot express at all.

### 52.6 Verdict

**Slang fits Shape 1 cleanly and leaves `processOne`/`processBulk` exactly as they are.** Its value, ranked
honestly:

1. multi-stage / chunked kernels become expressible on shader backends (today: impossible)
2. one GPU body instead of GLSL + hand-written WGSL
3. reflected settings bind automatically at any type (today: hand-listed floats)
4. SPIR-V / CUDA / Metal for free

**It does not reduce the two-bodies problem** (C++ for CPU, Slang for GPU). Nothing does, on shader
backends — §51.8: even a perfect C++ → WGSL compiler would not restore the block-as-functor model, because
Logical addressing forbids the mirrored block object. **The two-bodies property is a consequence of the
residency model, not of the language choice, and it is the thing to state plainly rather than keep trying to
engineer away.**

**Cost:** one build-time dependency (`slangc`), and blocks that want GPU acceleration on shader backends
write their kernel in Slang instead of GLSL. Neither is larger than what ships today; item 3 makes the block
author's job smaller.

## 53. IS OpenGL/WebGPU WORTH IT? — the FFT litmus test, quantified (2026-08-24)

Maintainer's hypothesis: _FFTs are only faster on the GPU (even natively) when N is very large or the data
is batched, to amortise the copy costs; most basic processing is slower in a browser anyway._
**Confirmed, and the numbers are more decisive than the hypothesis.**

All figures from the authoritative sweep (`reference_fft_backend_benchmark_table`, 2026-08-22, idle card,
clock pinned 1500 MHz, best of 3). Each backend is scored against **its own CPU baseline** — native GPUs vs
native `SimdFFT`, browser WebGPU vs in-browser `SimdFFT` — because that is the choice a user actually faces.

### 53.1 Speed-up vs the relevant CPU baseline (>1.00 = the GPU wins)

**native GLSL:GPU — wins 2 of 21 cells, best 1.56×**

| batch | 1024 | 2048 | 4096     | 8192     | 16384 | 32768 | 65536 |
| ----- | ---- | ---- | -------- | -------- | ----- | ----- | ----- |
| x1    | 0.01 | 0.07 | 0.11     | 0.11     | 0.22  | 0.24  | 0.36  |
| x16   | 0.30 | 0.66 | **1.56** | 0.49     | 0.62  | 0.48  | 0.54  |
| x128  | 0.25 | 0.52 | 0.98     | **1.56** | 0.48  | 0.53  | 0.43  |

**native SYCL:GPU — wins 17 of 21, every cell at batch ≥ 16, best 6.42×**

| batch | 1024 | 2048 | 4096 | 8192 | 16384    | 32768 | 65536 |
| ----- | ---- | ---- | ---- | ---- | -------- | ----- | ----- |
| x1    | 0.07 | 0.15 | 0.31 | 0.46 | 1.03     | 1.61  | 2.60  |
| x16   | 1.13 | 2.31 | 4.11 | 4.55 | **6.42** | 4.92  | 5.98  |
| x128  | 3.08 | 4.27 | 5.16 | 5.21 | 5.08     | 4.87  | 4.97  |

**browser WebGPU — wins 11 of 21, clean region at batch ≥ 16 and N ≥ 4096, best 3.75×**

| batch | 1024 | 2048 | 4096 | 8192 | 16384    | 32768 | 65536 |
| ----- | ---- | ---- | ---- | ---- | -------- | ----- | ----- |
| x1    | 0.01 | 0.08 | 0.13 | 0.16 | 0.36     | 0.75  | 1.12  |
| x16   | 0.29 | 0.61 | 1.20 | 2.33 | **3.75** | 1.29  | 1.78  |
| x128  | 0.48 | 1.02 | 0.76 | 1.26 | 2.02     | 2.37  | 2.67  |

### 53.2 The finding that decides the OpenGL question

**Native GLSL wins 2 cells out of 21, and both are isolated spikes.** At x16 it reads 1.56× at N=4096 and
0.49× at N=8192; at x128, 1.56× at N=8192 and 0.48× at N=16384. Its two wins are each surrounded by losses,
and the sweep independently flags GLSL as erratic across neighbouring cells. **That is not an operating
region — it is noise that occasionally crosses 1.0.** No product can be built on it.

Meanwhile SYCL, on the same card, wins every cell at batch ≥ 16, peaking at 6.4×. **Where a SYCL toolchain
exists, native GL compute has no performance case at all.**

### 53.3 The finding that saves the WebGPU question

WebGPU scores far better against _its_ baseline (11/21, up to 3.75×) than GLSL does against _its_ (2/21, up
to 1.56×) — **and the reason is not that WebGPU is a better backend. It is that the browser's CPU path is
handicapped.** WASM costs the CPU FFT ~3× (native SimdFFT ~15 G, in-browser ~5 G, reproduced twice). The GPU
therefore has a much easier target to beat in the browser than on the desktop.

**This inverts the intuition and is the single most useful conclusion here: GPU offload is worth _more_ in
the browser than natively-via-GL, precisely because the browser CPU is weak.** And in the browser there is
no alternative — AdaptiveCpp cannot be shipped to a browser, and WebGL2 (GLES 3.0) has no compute shaders at
all. WebGPU is the only GPU path that exists there.

The maintainer's other claim also holds exactly: at N ≤ 2048 **or** batch = 1, WebGPU loses everywhere.
**It is a targeted accelerator for compute-dense blocks, not a general one.**

### 53.4 The transfer budget — why architecture, not kernel speed, decides this

At N=4096, per sample:

|                                                                     | ns/sample | vs CPU          |
| ------------------------------------------------------------------- | --------- | --------------- |
| CPU `SimdFFT`                                                       | 3.90      | 1.0×            |
| GPU compute only (SYCL x128, 79.4 G)                                | 0.76      | **5.2× faster** |
| GPU + **naive** shared-USM readback (38 ns/sample, measured)        | 38.76     | **10× SLOWER**  |
| GPU + **staged** pinned transfer (0.75 ns/sample @32 MiB, measured) | 1.51      | **2.6× faster** |

**A naive host round trip does not erode the 5.2× kernel advantage — it inverts it into a 10× loss.** The
transfer is ~50× the compute time at this size. Everything the branch achieves rests on not paying it.

### 53.5 The FIR litmus test

Overlap-save FIR uses N ≥ 2M for an M-tap kernel, typically N = 4M, and batches over channels or successive
blocks. Mapping the regions above:

| filter kernel | FFT size        | native SYCL           | native GLSL            | browser WebGPU           |
| ------------- | --------------- | --------------------- | ---------------------- | ------------------------ |
| ≤ 256 taps    | N ≤ 2048        | 1.1–4.3× (batch only) | **loses**              | **loses**                |
| ~1024 taps    | N = 4096–8192   | **4.1–5.2×**          | 1.56× once, else loses | 1.2–2.3× at x16          |
| ~4096 taps    | N = 16384–32768 | **4.9–6.4×**          | loses                  | **3.75×** at x16/N=16384 |

**So the useful GPU envelope for FIR is N ≥ 4096 with batch ≥ 16** — kernels of roughly a thousand taps or
more, processed in batches of at least sixteen blocks. That is a real DSP regime (channelisers, matched
filters, pulse compression), not a corner case. Below it, the CPU wins on every backend.

**And FIR is exactly the chain the SubGraph exists for.** Overlap-save is `FFT → multiply → IFFT`: three
blocks, two interior edges. Without elision that is 6 host transfers; with it, 2. Given §53.4, that ratio is
the difference between the GPU being 2.6× faster and 10× slower. **The litmus-test use case is the one that
most needs the vehicle.**

### 53.6 Worth, by angle

| angle                                                   | verdict                                                                                                                                                                                                                   |
| ------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **native GL as a performance feature**                  | **No.** 2/21 cells, erratic, best 1.56×, and SYCL does 6.4× on the same hardware. Do not present GL as native GPU acceleration.                                                                                           |
| **native GL as a no-SYCL fallback**                     | Weak. It barely beats a good SIMD CPU FFT, and loses to it in 19 of 21 cells.                                                                                                                                             |
| **native GL as the dev/CI vehicle for the shader path** | **Yes — this is its real value.** It lets the shader-authoring path be developed, tested and CI-gated natively without a browser. That is a _maintainer_ benefit, not a user benefit, and it should be described as such. |
| **browser WebGPU**                                      | **Yes.** 2–3.75× in a clean region, and it is the _only_ GPU path a browser has. The weak WASM CPU baseline is what makes the offload pay.                                                                                |
| **the SubGraph / elision architecture**                 | **Essential, not optional.** §53.4: without it the kernel advantage inverts. This is the strongest justification the vehicle has.                                                                                         |
| **maintenance**                                         | Two shader backends sharing one residency model and (with Slang) one kernel source. Acceptable — but only if GL's role is honestly stated as dev/CI rather than performance.                                              |

### 53.7 The gap this analysis cannot close

**There is no in-graph GLSL or WebGPU FFT number anywhere.** Every shader-backend figure above is
_raw algorithm_. The in-graph measurements that exist are CPU (`bm_fft_stream_graph`: framework costs
1.2–1.4×) and `gpu:sycl` edge-buffer throughput. Given that the framework costs 1.2–1.4× _and_ §53.4 shows
the transfer dominating by ~50×, **the in-graph shader numbers could be far worse than the table implies,
and nothing here would have caught it.**

Concretely missing: `FFT<float>` stream mode in a real graph on `gpu:glsl`, and the same in-browser on
WebGPU, both inside a SubGraph so the elision is active, compared against the in-graph CPU baseline
(198 MS/s at N=1024, 185 at N=4096, 171 at N=16384).

**Until that exists, §53.6's "yes" for browser WebGPU rests on raw-algorithm numbers plus a measured
transfer model — a sound inference, not an end-to-end measurement.** State it that way in the PR.

## 54. CONFIRMED — no streaming-DSP win from WebGPU/WebGL; visualisation is the real case (2026-08-24)

Maintainer asked for confirmation that FFT/FIR/channeliser work sees no performance win on WebGPU/WebGL.
**Confirmed for the streaming regime, with one boundary that must be stated or the claim is wrong.**

| backend                                    | verdict                                                                                                                                                                                                                                          |
| ------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **WebGL**                                  | **Confirmed absolutely** — WebGL2 is GLES 3.0 and has **no compute shaders at all**. There is nothing to execute; the performance question does not arise.                                                                                       |
| **native OpenGL compute**                  | **Confirmed** — 2 wins in 21 cells, both isolated spikes with losses either side (§53.1). Not an operating region.                                                                                                                               |
| **browser WebGPU at batch = 1**            | **Confirmed** — loses every cell except N=65536 at 1.12×, inside noise. For a single-channel chain doing one transform per `work()` call there is no win.                                                                                        |
| **browser WebGPU at batch ≥ 16, N ≥ 4096** | **NOT confirmed — there IS a real 2–3.75× win.** Reachable in a flowgraph by letting the edge buffer fill, at a latency cost (128 transforms at N=4096 ≈ 524k samples ≈ 52 ms at 10 MS/s). Fine for spectrum monitoring, not for a control loop. |

**Channeliser specifically:** a 64–1024-channel polyphase bank sits _below_ the crossover and loses. Only
long matched filters / pulse compression at N ≥ 4096 across many channels win.

### 54.1 Visualisation is different, and for a principled reason: rendering has no readback

§53.4 showed the readback costs **38 ns/sample against 0.76 ns of compute** — that single term is what
inverts a 5.2× win into a 10× loss. **For visualisation the result is pixels, consumed by the display
pipeline on the device. The data never comes back, so the term that kills compute offload does not exist.**

Waterfalls, spectrograms and constellation displays are additionally embarrassingly parallel and
shader-native — the workload shaders were designed for, rather than one retrofitted onto them.

**And the combination is the compelling case:** an FFT running on the GPU _feeding_ a GPU visualisation is
viable even though the FFT alone is not. Input samples upload once; magnitude and render happen on-device;
nothing downloads. One transfer instead of six. `FFT → magnitude → waterfall` is exactly the elision chain —
and the value is **not "accelerate the DSP" but "keep the display path off the CPU"**.

---

## 55. RE-EVALUATION AGAINST THE ORIGINAL PREMISE (2026-08-24)

**Maintainer decision D5: WebGPU/GLSL move OUT of GR4 core and INTO the visualisation sinks. Core supports
CPU and SYCL only (later possibly CUDA/ROCm).** This is the largest scope change on the branch and it
invalidates or demotes several earlier conclusions. Re-evaluated below against §2's original goals.

### 55.1 The finding that makes D5 cheap — and it is bigger than the file moves

**Under CPU + SYCL only, the entire chain-binding / epoch machinery becomes unnecessary — because SYCL never
needed it.** Three pieces of existing evidence, none of which was assembled in this order before:

1. **§30.1 (measured, committed `7a7eae52`):** _"an interior edge really is device memory"_. On this machine
   a SYCL interior edge resolves to `CudaVmmMemoryResource` — device-only memory, verified dereferenceable
   by the resolved context, not merely "different from default".
2. **`ExecutionStrategy.hpp:476-477, 569-570`:** per-block dispatch already probes
   `ctx.isDeviceAccessible(span.data())` and skips both copy and scratch when the edge is device memory —
   _"a device→device edge already holds USM the kernel can read/write in place… This keeps data on the
   device from one block to the next."_
3. **§31 (the elision that was built):** its only readers are the GLSL hatch (`fft.hpp:194-227`) and the
   WebGPU hatch (`fft.hpp:275-304`). The SYCL hatches (`fft.hpp:140`, `:401`) never touch it — the first
   does `q.memcpy(...)` and transforms in place on the edge span. `ExecutionStrategy.hpp` has **zero**
   references to the binding API. The 34-vs-66 measurement is a **GLSL** measurement.
   **CORRECTED (fable review, 2026-08-24):** the machinery does NOT live on `DeviceContextGLSL` — commits
   `6d7527d1` and `0a754155` moved all of it (`DomainBinding`, `bindDomainBuffer`, `domainBinding`,
   `beginDomainEpoch`, `stampDomainChain`, `domainChainHolds`, `pooledChainBuffer` and their maps) onto the
   **base `DeviceContext`** (`DeviceContext.hpp:87-236`), a file that **stays**. Removal is surgery inside a
   staying header, not deletion of a leaving file.

**Conclusion: interior residency on SYCL comes from the edge PMR, per block, with no SubGraph involved. The
chain-binding exists because GL/WebGPU hand out opaque handles that cannot live in a `CircularBuffer`.**

**Two corrections from the fable review, both strengthening the case:**

- **The machinery is structurally unusable on SYCL, not merely unused.** `pooledChainBuffer`
  (`DeviceContext.hpp:159-171`) hardwires `Residency::opaque`, and `DeviceContextSycl::allocate` **refuses**
  it (`:189` "SYCL backend has no opaque tokens", `:204`). A hypothetical SYCL consumer would get an invalid
  buffer on every call.
- **A SYCL chain IS bound today, inertly.** `bindChainBuffers()` (`DeviceDomain.hpp:188-239`) is
  backend-agnostic: any chain sharing a resolvable non-host `compute_domain` — `gpu:sycl` included — gets
  bindings, and `Domain::work()` pays `beginDomainEpoch` (mutex + map op) **per call** (`:114-116`) for
  nothing. So "SYCL never _needed_ it" is right, but "it never engages for SYCL" would be false — and
  **removal also deletes real per-`work()` overhead.**
- **Careful with "its consumers are leaving":** the consumers are hatches embedded in **staying** files
  (`fft.hpp`; `dispatchGlsl` consuming `shaderFragment()` at `ExecutionStrategy.hpp:330`). True of the code,
  false of the files.

**Stronger evidence than §30.1 exists and should be cited instead:** `core/test/qa_DeviceResidency.cpp:44-69,
91-114` is a committed **flat-graph** test — host source → gpu Gain → gpu Gain → host sink in a plain
`gr::Graph` — asserting the middle edge is device-only, double-mapped and numerically correct. The mechanism
is flat-`Graph` code (`Graph.hpp:721-758`), no Domain anywhere.

This also aligns with the already-recorded streaming finding: _fix the host BOUNDARY (pinned USM + real
`queue.memcpy` in `HostToDevice`/`DeviceToHost`), not interior edges_ — the 38 → 0.75 ns/sample win. **That
work is still outstanding and is now the single highest-value performance item on the branch.**

### 55.2 What leaves core

| what                                                                                                                                                          | lines                                                 |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------- |
| `core/device/`: `DeviceContextGLSL`, `GlComputeContext`, `DeviceContextWebGpu`, `WebGpuRuntime`, `GlslRuntime`, `GLSL2WGSL`, `ShaderFusion`, `ShaderFragment` | **1,269** (30 % of `device/`)                         |
| `algorithm/`: `GlslFFT.hpp` (214) + `WgslFFT.hpp` (281)                                                                                                       | **495**                                               |
| `tools/webgpu/` (harness, benchmarks, browser runner)                                                                                                         | **2,051**                                             |
| chain-binding/epoch API in `DeviceContext` + `Domain::bindChainBuffers`                                                                                       | ~**32 call sites**                                    |
| `Residency::opaque` — `DeviceContextSycl.hpp:204` explicitly _rejects_ it; it exists only for GL/WebGPU handles                                               | enum arm + the `devicePointer<T>()` refusal machinery |
| `ExecutionStrategy::dispatch` shader arms: `dispatchGlsl`, `dispatchGlslBulk`, `dispatchWebGpuBulk`                                                           | 3 of 4 backend arms                                   |
| tests: `qa_FFTDomain` GLSL leg, `qa_DeviceContext` (GLSL2WGSL/ShaderFusion), the `webgpu-browser` CI lane                                                     | —                                                     |

**CORRECTED (fable review): the file arithmetic is exact but it is an UNDERCOUNT.** 3,815 lines of whole
files leave. On top of that, ~400-500 lines must be excised **in place from staying files**, none of it in
the 3,815:

| staying file                                         | what comes out                                                                                                                                                                                   | ~lines                 |
| ---------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------- |
| `blocks/fourier/fft.hpp`                             | the `GR_DEVICE_HAS_GL_COMPUTE` and `GR_DEVICE_HAS_WEBGPU` regions, `processBulk_glsl`, `_glslFft`                                                                                                | ~160                   |
| `core/device/ExecutionStrategy.hpp`                  | `dispatchGlslBulk` (:273), `dispatchWebGpuBulk` (:291), `dispatchGlsl` (:315), the two `Has*BulkForSpans` concepts (:84-110), the switch arms (:211-224), the `ShaderFragment.hpp` include (:29) | ~150                   |
| `core/device/DeviceContext.hpp` + `DeviceDomain.hpp` | the chain/epoch machinery (`:87-236`) and `linearChains`/`bindChainBuffers`/`debug_fill_host_rings`                                                                                              | ~250                   |
| **`blocks/basic/CommonBlocks.hpp`**                  | **`:15` includes `ShaderFragment.hpp` and `builtin_multiply` exposes a `shaderFragment()` hatch at `:44`** — a staying _basic blocks_ header depends on a leaving file                           | small but load-bearing |

`HasShaderFragment` also participates in the shared dispatch-eligibility predicate (`:165`, `:199`), so this
is an edit to shared logic, not just arm deletion. **Blast radius also under-scoped:** at least 7 staying
test/bench files need their GLSL/WebGPU legs excised — `qa_DeviceBlockStyles`, `qa_DevicePerformance`,
`device_test_helpers.cpp`, `qa_FFTDevice`, `qa_FFTPerformance`, `bm_FFT_backends[_helpers]`. And
`qa_DeviceContext` does **not** leave wholesale: its first suite (`:19-119`, base/CPU context + `parallelFor`)
stays; only the `DeviceContextGLSL` suite (`:120+`) goes.

**Also undersold:** `ExecutionStrategy` does not reduce to "one arm plus CPU fallback" — the two framework
tiers (`dispatchDeviceBulk` :445, `dispatchAutoParallel` :535) stay as well, both SYCL-only in effect.

### 55.3 Goals re-scored under D5

| #   | goal                                          | before            | after D5                                 | why it moved                                                                                                                                                                    |
| --- | --------------------------------------------- | ----------------- | ---------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | preserve the Block API                        | MET               | **MET**                                  | unchanged                                                                                                                                                                       |
| 2   | `const noexcept processOne` auto-parallelises | MET               | **MET, and now honestly so**             | the auto-parallel path only ever worked on SYCL (`ParallelFor.hpp` has a device path for SYCL alone). With shader backends gone the claim stops carrying an implicit exception. |
| 3   | backend escape hatches, zero-cost             | MET               | **MET, simplified**                      | one hatch family (`processBulk_sycl`, later `_cuda`) instead of two. `shaderFragment()` leaves core entirely.                                                                   |
| 4   | selection via `compute_domain`                | MET               | **MET**                                  | unchanged; grammar already covers `gpu:cuda`, `gpu:hip`                                                                                                                         |
| 5   | zero cost for CPU-only graphs                 | MET (+64 B/block) | **MET, improved**                        | fewer headers, one dispatch arm; the +64 B stays (one-layout ODR decision)                                                                                                      |
| 6   | CI-testable without GPU hardware              | MET               | **PARTIAL — two regressions, see §55.7** | "the acpp OMP lane suffices" is WRONG as written: the no-silent-skip guard is derived only for GL, and D5 removes the only sanitized device lane                                |
| 8   | composition over modification                 | MET               | **MET, more strongly**                   | ~3,800 fewer lines around the same seam                                                                                                                                         |

**Use-case A** (standalone device blocks): met, SYCL. **Use-case B** (data stays on device between blocks):
**met by the edge PMR, per block, measured** (§30.1) — and _not_ by the SubGraph, which was the shader-path
mechanism.

### 55.4 The question D5 forces: does the SubGraph still earn its keep?

**Its headline justification is gone.** §53.5 argued the FIR chain `FFT → multiply → IFFT` most needs the
vehicle — but that argument was about **GLSL/WebGPU transfers**. On SYCL those interior edges are already
device memory without it.

What genuinely remains:

| remaining value                                                                | strength                                           |
| ------------------------------------------------------------------------------ | -------------------------------------------------- |
| one queue submission for a whole chain, no per-block host sync                 | real, **unmeasured**                               |
| avoids per-block `compute_domain` re-resolution + registry lookup per `work()` | real, small (already cached in `schedulerCache`)   |
| explicit grouping vehicle the user selects (D3)                                | real, but `Graph::groupBlocks` now covers grouping |
| synchronous ordered execution on the caller's thread                           | real, and unique                                   |

**This needs an honest decision, and I am not taking it here.** The options are (a) keep the SubGraph as a
_scheduling_ construct with its device-elision justification withdrawn, (b) defer it entirely and ship
per-block SYCL dispatch, which is measured and sufficient for use-case B, or (c) keep it but re-justify it
on the one-queue-submission win — **which would first have to be measured, and has not been.**

Given D1 committed to a SubGraph-owns-dispatch rewrite, and D5 removed that rewrite's main beneficiary,
**D1 should be revisited before any of it is built.**

### 55.5 Everything else, re-scored

| item                                                         | status under D5                                                                                                                                                                                |
| ------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Slang** (§51, §52)                                         | **DROP from core.** Its value was collapsing GLSL + WGSL into one GPU body. Core no longer has shader backends. It may matter to the visualisation sinks — that is their decision, not core's. |
| `shaderFragment()` reclassification (§50.2)                  | **moot for core** — moves with the shader backends                                                                                                                                             |
| C++ → WGSL / SPIR-V spike (§51.9)                            | **DROP** — no core consumer                                                                                                                                                                    |
| GLSL → WGSL transpiler decision (§51.3)                      | **moves to the visualisation sinks**                                                                                                                                                           |
| `GLSL2WGSL.hpp`, `ShaderFusion.hpp`                          | leave core with the rest; no longer a delete-or-keep argument                                                                                                                                  |
| **host-boundary staging** (pinned USM + real `queue.memcpy`) | **PROMOTED to top priority** — §55.1; the 38 → 0.75 ns/sample win is now the branch's main performance item                                                                                    |
| `DeviceLog`                                                  | unchanged — **IN, first, two commits**                                                                                                                                                         |
| `GR4_REQUIRE_DEVICE`, fallback loudness (§50.5)              | unchanged                                                                                                                                                                                      |
| `compute_domain`                                             | unchanged — grammar already admits `gpu:cuda` / `gpu:hip`                                                                                                                                      |
| CUDA / ROCm                                                  | **future extension points, and now the natural next backends** — both are pointer-based like SYCL, so they reuse the residency model rather than needing the opaque path                       |

### 55.6 Honest summary of the branch under D5

**What it becomes:** a CPU + SYCL device-execution layer for GR4 — one dispatch seam in `Block.hpp`, one
backend arm, residency by edge PMR (measured), a device diagnostics channel, and a loud CPU fallback.
Roughly **3,800 fewer lines in core** than the branch carries today, with the shader work relocated to
visualisation where §54.1 shows it actually pays.

**What it gives up:** GPU compute in the browser. Per §54 that costs little in the streaming regime the
framework targets — WebGPU loses at batch = 1 across the board — and what remains valuable there
(display-path offload) is precisely what moves to the sinks.

**What still needs doing before the premise is fully met:** the host-boundary staging (§55.1), and a
decision on the SubGraph (§55.4).

**This is a better branch than the one reviewed on 2026-08-22.** The review's cost-benefit case rested on
9 modified core headers against a large additive device layer; D5 removes ~30 % of that additive layer and
one of the two things that made the dispatch path complex. The premise — _write `processOne`/`processBulk`
once, run it on a device_ — is served more honestly by one good backend than by four of unequal quality.

### 55.7 CORRECTIONS FROM THE ADVERSARIAL REVIEW (fable, 2026-08-24)

Four claims attacked. Claims 1 and 3 confirmed; 2 and 4 partially wrong. Corrections are folded into §55.1,
§55.2 and §55.3 above. The two findings that change decisions:

#### 55.7.1 After D5 the SubGraph has ZERO in-tree production users

**The only non-test consumer of `makeDomain`/`Domain` anywhere in the repo is
`tools/webgpu/fft_graph/main.cpp` — which leaves under D5.** Everything else that exercises the SubGraph is
its own test suite.

This is decisive for §55.4. Combined with the finding that SYCL interior residency is already delivered by
the edge PMR in a **flat graph** (`qa_DeviceResidency.cpp:44-69`), option **(b) — defer the SubGraph
entirely and ship per-block SYCL dispatch — is now the strongly indicated choice**, and D1
(SubGraph-owns-dispatch) should not be built until something needs it. The vehicle was constructed for a
backend family that is leaving.

#### 55.7.2 D5 removes the only sanitized device lane, and the no-silent-skip guard

Two concrete CI wires break, and §55.3's "MET, cheaper" for goal 6 was wrong:

1. **`GR4_REQUIRE_DEVICE` is derived only for GL.** `CMakeLists.txt:329-335` sets it from
   `if(GR_HAS_GL_COMPUTE) → gpu:glsl`. There is **no `gpu:sycl` derivation**, although the tests already
   honour one (`qa_DeviceDomain.cpp:205, 211`). Remove the GL lane and the acpp lane can skip every device
   test — failed `registerSyclRuntime`, unresolved `gpu:sycl` — and still pass green. **That is exactly the
   regression §45 was built to prevent.** Fix is small: derive `GR4_REQUIRE_DEVICE=gpu:sycl` when SYCL is
   enabled. It must land with D5, not after.
2. **All sanitizer coverage of live device dispatch disappears.** `GL-llvmpipe` (`ci.yml:85-89`) is the only
   ASan lane that executes a real backend through the dispatch machinery. The AdaptiveCpp lane runs with
   sanitizers **off** (`ci.yml:98`), and that is structural, not incidental: `CMakeLists.txt:355` declares
   SYCL incompatible with ASan/UBSan/TSan. **After D5 no sanitized lane exercises any dispatching device
   path** — on a branch whose history contains two device-adjacent use-after-frees. This is a genuine cost
   of D5 and must be stated in the PR rather than discovered later. Mitigation options: keep a
   CPU-`DeviceContext` sanitized lane that still traverses `ExecutionStrategy`, or accept the gap explicitly.

#### 55.7.3 Also noted

**The load-bearing residency mechanism has never been CI-visible.** `qa_DeviceResidency`'s flat-graph
device-only test gates on `deviceOnlyResourceFor(-1) != nullptr` (CUDA driver present) and silently returns
otherwise (`:93-94`). No GitHub runner has that, before or after D5. Not a D5 regression — but §30.1/§55.1's
central mechanism is verified **only on the maintainer's machine**, and goal 6's "CI-testable without GPU
hardware" does not cover it. Say so in the PR.

## 56. D5 EXECUTION PLAN — decisions and commit map (2026-08-24)

### 56.1 Decisions taken

| #       | decision                                                                                                                                                              |
| ------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **D5**  | **Drop the WebGPU and GLSL backends entirely** from GR4 core. Shader work relocates to the visualisation sinks (separate effort). Core = CPU + SYCL, later CUDA/ROCm. |
| **D6**  | **Rebuild history** on `origin/main`, same branch name, didactic commits gated by functionality and unit-testability. Target **~33**, hard ceiling 50.                |
| **D7**  | **Rebuild first, phase 5 on top** — restructure existing work into reviewable history before writing the SubGraph rewrite.                                            |
| **D8**  | **Host-boundary staging IS in scope** (pinned USM + real `queue.memcpy`): the 38 → 0.75 ns/sample fix.                                                                |
| **D9**  | **Add a sanitized CPU-`DeviceContext` CI lane** to replace the ASan coverage lost with GL-llvmpipe.                                                                   |
| **D10** | **Both author and committer dates backdated** to the original development window (2026-07-09 … 2026-08-22). New phase-5 work carries its own date.                    |

**Snapshot taken before any rewrite:** branch `backup/pre-d5-rewrite-20260824` and tag
`snapshot/webgpu-glsl-complete-20260824`, both at `6387faed`. The WebGPU/GLSL work — 3,815 lines, measured
and browser-verified — is recoverable from these refs and from them only, once history is rewritten.

### 56.2 SubGraph hypothesis — CROSS-CHECKED, CONFIRMED, and §55.4 option (b) WITHDRAWN

The maintainer's claim: SubGraph-owns-dispatch is semantically the better mental model and more future-proof
than per-block dispatch with auto-domain-aggregation. **Confirmed — on grounds neither of us had stated.**

**D3 ("no auto-detection") entails the SubGraph.** The workflow requires automatic insertion of
`HostToDevice`/`DeviceToHost` at the device boundary. Those blocks exist
(`blocks/basic/TransferBlocks.hpp:13, 68`) but nothing inserts them — zero references in `Graph.hpp`.

- In a **flat** graph with per-block `compute_domain`, inserting them requires _inferring_ where the domain
  changes — exactly the hidden-assumption auto-detection D3 rejected.
- In a **SubGraph**, the boundary is _declared_ by membership; transfers go at the exported ports. No
  inference.

**So the SubGraph is the only construct that makes automatic transfer insertion compatible with D3.** Two
further supports: one queue per SubGraph generalises directly to a CUDA or HIP stream, where a scattered set
of per-block domains does not.

**§55.7.1's "zero production users → defer it" is WITHDRAWN.** That finding was correct about the _elision_
justification, which is dead. The _composition_ justification is different, live, and stronger. The two are
not in conflict and §55.4 should be read as: option (a), keep the SubGraph, elision justification withdrawn,
composition justification substituted.

### 56.4 Commit map — target ~33

> **EXECUTED, and the target was wrong by half.** The rebuild landed 21 commits (§58.5), the maintainer judged
> that still too many, and the regroup to 10 (§58.7) plus phase 5 gave the **16** the branch has today. The map
> below is what was planned, kept because the phase grouping is still how the commits are ordered.

| phase                         | n   | commits                                                                                                                                                                                                                    |
| ----------------------------- | --- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 0 · pre-existing `main` fixes | 2   | double-mapped munmap leak · moved-block memory resource                                                                                                                                                                    |
| 1 · compute-domain vocabulary | 3   | `ComputeDomain` grammar + registry · `compute_domain` on `Block` · edge PMR precedence                                                                                                                                     |
| 2 · device runtime            | 4   | `DeviceContext` + `Residency` + `DeviceBuffer` · USM resource + SYCL runtime + registry · device-only double-mapped memory · device error channel                                                                          |
| 3 · dispatch seam             | 5   | one `Block` layout · `ExecutionStrategy` + the seam · `DeviceRelocatable` + mutation guards · auto-parallel `processOne` · `processBulk_sycl` + N-ary ports                                                                |
| 4 · kernel bodies             | 3   | pre-allocated callable bodies · framework device bulk · kernel-facing views                                                                                                                                                |
| 5 · **SubGraph** (new work)   | 8   | vehicle + rename · owns-dispatch rewrite · **automatic transfer insertion** · boundary naming + refusals · **vertical test: host→device → chain → device→host** · lifecycle/quiescence · **host-boundary staging (D8)** ×2 |
| 6 · FFT                       | 3   | SYCL FFT · one FFT block · spectrum stages in a kernel                                                                                                                                                                     |
| 7 · diagnostics               | 2   | `DeviceLog.hpp` (std-lib only, lands first) · logger backend                                                                                                                                                               |
| 8 · tests / CI / docs         | 4   | test helpers + tag/settings pinning · `GR4_REQUIRE_DEVICE` incl. **`gpu:sycl` derivation** · CI lanes incl. **sanitized CPU-context lane (D9)** · docs                                                                     |

**Total 34.** Phases 0–4, 6, 7 are restructuring; phase 5 is new development, landing after the rebuild
(D7). Phase 8 partly new (the two CI gaps §55.7.2).

### 56.5 Carried into the work, not to be lost

- `blocks/basic/CommonBlocks.hpp:15` includes `ShaderFragment.hpp`; `builtin_multiply` exposes a
  `shaderFragment()` hatch at `:44` — must be de-shadered with the drop.
- `GR4_REQUIRE_DEVICE` is derived only for GL (`CMakeLists.txt:329-335`); the `gpu:sycl` derivation must
  land **with** the drop or device lanes can pass green having skipped everything.
- `qa_DeviceContext` splits: `:19-119` (base/CPU context, `parallelFor`) stays, `:120+` (GLSL suite) goes.
- Still open, unscheduled: `kBlobAlignment` 16-vs-8
  blocking the by-value tag route.
- `qa_DeviceResidency`'s device-only test gates on a CUDA driver and silently returns otherwise
  (`:93-94`) — the load-bearing residency mechanism is verified only on the maintainer's machine.

## 57. D5 DROP — EXECUTED (2026-08-24, branch `d5-drop-shaders`)

Step 1 of §56.10. The WebGPU/GLSL removal is complete in the tree; the history rebuild (D6) follows.

### 57.1 What was removed

**23 whole files, 4,800 lines** — the 8 shader headers in `core/device/`, `GlslFFT.hpp` + `WgslFFT.hpp`,
all of `tools/webgpu/`, and **`core/include/gnuradio-4.0/execution/`** (D11).

**In-place surgery on staying files** (the undercount §55.2 was corrected for):

| file                          | change                                                                                                      |
| ----------------------------- | ----------------------------------------------------------------------------------------------------------- |
| `ExecutionStrategy.hpp`       | 655 → 496 lines; the four-arm backend `switch` became one guarded `if constexpr`                            |
| `DeviceContextRegistry.hpp`   | **the parallel `_schedulers` map deleted**; `tryResolve()` now returns `DeviceContext*`                     |
| `Block.hpp`                   | the cached scheduler pointer became `device::DeviceContext* _deviceContext`; forward-decl gone              |
| `BlockTraits.hpp`             | `HasShaderFragment` deleted; `DeviceEligible` is now exactly `AutoParallelisable`                           |
| `BackendDetect.hpp`           | `DeviceBackend` enum: `{SYCL, GLSL, CUDA, ROCm, WebGPU, CPU_Fallback}` → `{SYCL, CUDA, ROCm, CPU_Fallback}` |
| `fft.hpp`                     | both hatch regions + both shader FFT members removed                                                        |
| `CommonBlocks.hpp`            | `builtin_multiply`'s `shaderFragment()` hatch removed (§55.7's staying-header dependency)                   |
| `CMakeLists.txt` ×5           | `GR_ENABLE_GL_COMPUTE` / `GR_ENABLE_WEBGPU` options and every `GR_HAS_*` block                              |
| 9 test/bench files            | GLSL/WebGPU legs excised; two obsolete suites deleted outright                                              |
| `docs/USER_API_GPU_Blocks.md` | rewritten: two styles, not three                                                                            |

**The cached scheduler really was "a `DeviceContext*` in a costume"** — the audit's phrase, now
proven: removing it deleted an entire parallel map alongside `_contexts` and cost nothing.

### 57.2 The two CI decisions, landed with the drop

- **D9 sanitized lane**: `GL-llvmpipe` → **`device-CPU-asan`** (`ci.yml`), same ASan coverage over
  `ExecutionStrategy` / SubGraph / lifecycle via the CPU `DeviceContext`. SYCL cannot be sanitized
  (`CMakeLists.txt`), which is exactly why this lane must exist.
- **`GR4_REQUIRE_DEVICE`**: derivation moved from `GR_HAS_GL_COMPUTE`→`gpu:glsl` to
  `GR_USE_ADAPTIVE_CPP`→**`host:sycl`**. `host:sycl` and NOT `gpu:sycl`: `registerSyclRuntime()` publishes a
  domain per device _kind that exists_, so a GPU-less runner has only the CPU one. Requiring the domain that
  is present whenever SYCL comes up at all is what makes this a skip-detector rather than a GPU-presence
  test. (Also fixed a stale doc comment in `SyclRuntime.hpp` that said `cpu:sycl`; the code says `host`.)
- The `webgpu-browser` job is gone.

### 57.3 Verification

`build-gcc15-debug` (gcc15, registry OFF): configure clean, **12 of 13 targeted tests pass**. The single
failure is `qa_Graph` / `Graph::groupBlocks` — **pre-existing and documented in §48.3**: those tests need a
scheduler typename from the block registry, which this dir has OFF. Not a regression.
`build-ci-clang20-debug` (clang20 libc++, registry ON): `gnuradio-core` and `gnuradio-blocklib-core` build
clean; fourier/core test targets in progress.

### 57.4 Trap hit and worth remembering

Cutting `dispatchGlsl` by scanning forward for the next `template<typename InputSpans, typename OutputSpans>`
**over-cut 82 lines of shared, backend-neutral helpers** that happened to sit after it —
`kOwnsDeviceShadow`, `deviceMirror`, `isFirstUseOfTheseSettings`, `autoParallelMutatesItsOwnState`,
`bulkMutatesItsOwnState`, `staleMirrorDiagnostic`, `runDeviceBulkCore`. The compiler caught it
(`'kOwnsDeviceShadow' was not declared`), but only because they were _used_. **When excising a region by
pattern, verify the end boundary against the symbol list, not against the next syntactic match.**

## 58. HISTORY REBUILD — mechanics and progress (2026-08-24, branch `syclExperiments-rebuild`)

Step 2 of §56.10. Target tree is `ee5f76f1` on the throwaway branch `d5-drop-shaders` (86 files,
**+11,204 / −563** against `origin/main`, down from 113 files / +17,662 before D5).

### 58.1 The mechanics that work

1. `git checkout <target> -- <paths>` for files that belong to exactly ONE phase — the vast majority
   (every new `device/*.hpp`, every new test).
2. **Craft an intermediate for files that span phases** (`Block.hpp`, `CircularBuffer.hpp`, the CMakeLists).
   **Base the intermediate on `HEAD`, never on `origin/main`** — see §58.3.
3. `git cherry-pick -n <sha>` where an original commit is already a clean single-purpose change (both
   phase-0 fixes were).
4. `GIT_AUTHOR_DATE=… GIT_COMMITTER_DATE=… git commit` — D10, both dates backdated.
5. `scratchpad/prune_tests.py <CMakeLists…>` after each checkout: drops `add_ut_test` / `add_library` /
   `add_executable` whose source does not exist yet, **and every later command mentioning a vanished
   target** (`target_link_libraries`, `set_tests_properties`, `add_test` with `$<TARGET_FILE:…>`). Without
   the last part an intermediate commit configures with dangling references.

### 58.2 Progress — 5 of ~34

> **SUPERSEDED by §58.5 (complete) and §58.7 (regrouped).** Progress log only.

| #   | commit                                                                          | date       | phase |
| --- | ------------------------------------------------------------------------------- | ---------- | ----- |
| 1   | `fix(core): keep a moved block's memory resource`                               | 2026-07-09 | 0     |
| 2   | `fix(core): release both halves of a double-mapped buffer`                      | 2026-08-07 | 0     |
| 3   | `feat(core): let a compute domain say how it executes, not just where it lives` | 2026-07-10 | 1     |
| 4   | `feat(core): let a memory resource declare what its memory can do`              | 2026-08-07 | 1     |
| 5   | `feat(graph): resolve an edge's memory resource by an explicit precedence`      | 2026-07-12 | 1     |

Each verified to configure and build `gnuradio-core` (+ its own test where one exists) before committing.

### 58.3 Trap, hit once — a crafted intermediate must be based on HEAD

Building an intermediate `Block.hpp` from `origin/main`'s version **silently reverted commit 1** (the
moved-block `_allocResource` fix, a single line). The build still passed, because nothing in the new commit
depended on it. Caught only by re-grepping for the earlier commit's own change.

**Rule: when crafting a partial file state, start from `git show HEAD:<path>` and add to it. Never start
from `origin/main` once any commit has already touched that file.** After crafting, grep for the marker of
every prior commit that touched the same file.

`origin/main` already carries `compute_domain`, `_computeDomainIsDevice` and `_deviceFallbackWarned` on
`Block` — part of this work merged upstream already, so phase 1 needed one commit fewer than §56.4 planned.

### 58.4 The chain/epoch machinery is GONE — §55.1 executed, and it was dead on arrival after D5

Checked before continuing the rebuild: after the shader drop the chain machinery had **zero readers**.
`domainBinding()`, `stampDomainChain()`, `domainChainHolds()` and `pooledChainBuffer()` had no callers at
all — those were the GLSL and WebGPU hatches. `Domain` still _wrote_ bindings nobody read and paid a mutex
plus a map operation **per `work()` call** for it, exactly as the fable review predicted.

Building a didactic history that introduces machinery with no reader would be perverse, so it never enters
the new history. Removed from the target tree:

| file                            | removed                                                                                                                                                                                                                                      |
| ------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `device/DeviceContext.hpp`      | `DomainBinding`, `bindDomainBuffer`, `clearDomainBindings`, `domainBinding`, `beginDomainEpoch`, `stampDomainChain`, `domainChainHolds`, `pooledChainBuffer`, `nextChainId`, and the five maps + mutex + atomic behind them (**−116 lines**) |
| `device/DeviceDomain.hpp`       | `linearChains()`, `bindChainBuffers()`, `_chains`, `_context`, `debug_fill_host_rings`, and the epoch bump in `work()` (**−121 lines**)                                                                                                      |
| `core/test/qa_DeviceDomain.cpp` | the cycle/chain-count test (−25)                                                                                                                                                                                                             |

`hostTransferCount()` / `hostUploadCount()` / `hostDownloadCount()` **stay** — they measure the boundary,
which is still the thing that matters (§53.4), and they are independent of chains.

**−262 lines**, all tests green (`qa_DeviceDomain`, `qa_DispatchGroup`, `qa_DeviceContext`, `qa_DeviceSeam`,
`qa_DeviceBlockStyles`). Target tree is now **`3c3fd663`**: 86 files, **+10,947 / −563** vs `origin/main`.

**Phase 5 reintroduces chain discovery** — under D1 the SubGraph scheduler is the thing that will actually
consume it, and `debug_fill_host_rings` returns with it. That is the didactic order: the mechanism arrives
with its reader, not before.

### 58.5 REBUILD COMPLETE — 21 commits, tree identical to target

`git rev-parse HEAD^{tree}` equals `3c3fd663^{tree}`: the rebuilt history reproduces the target tree
exactly, with **zero** remaining delta.

| #   | commit                                                                          | date       |
| --- | ------------------------------------------------------------------------------- | ---------- |
| 1   | `fix(core): keep a moved block's memory resource`                               | 2026-07-09 |
| 2   | `fix(core): release both halves of a double-mapped buffer`                      | 2026-08-07 |
| 3   | `feat(core): let a compute domain say how it executes, not just where it lives` | 2026-07-10 |
| 4   | `feat(core): let a memory resource declare what its memory can do`              | 2026-08-07 |
| 5   | `feat(graph): resolve an edge's memory resource by an explicit precedence`      | 2026-07-12 |
| 6   | `feat(device): a backend-neutral device runtime`                                | 2026-07-09 |
| 7   | `feat(core): let a stream buffer live in memory the host may never touch`       | 2026-08-07 |
| 8   | `feat(device): say which blocks may be copied into device memory, and why not`  | 2026-07-10 |
| 9   | `feat(core): let an ordinary block run on a device, unchanged`                  | 2026-07-09 |
| 10  | `feat(device): give the expert a hatch that owns the queue`                     | 2026-07-12 |
| 11  | `feat(device): logging from inside a kernel, rendered on the host`              | 2026-07-09 |
| 12  | `feat(basic): blocks that move samples across the host/device boundary`         | 2026-08-08 |
| 13  | `test(device): pin what device residency, auto-parallel and faults actually do` | 2026-08-07 |
| 14  | `test(core): pin the contract a synchronously-driven block group depends on`    | 2026-08-20 |
| 15  | `feat(device): run a group of blocks as one compute domain`                     | 2026-08-21 |
| 16  | `feat(algorithm): a SYCL Stockham FFT, and spectrum stages a kernel can call`   | 2026-07-11 |
| 17  | `feat(fourier): one FFT block for the host and device paths`                    | 2026-08-13 |
| 18  | `bench(fourier): measure the backends against each other, and in a real graph`  | 2026-07-12 |
| 19  | `build(device): make the SYCL backend an explicit build decision`               | 2026-08-08 |
| 20  | `ci(device): run the device path without a GPU, and refuse to skip it silently` | 2026-08-09 |
| 21  | `docs: how to write a block that runs on an accelerator`                        | 2026-08-13 |

**21, not the estimated 34** — three groups turned out to be single connected components that could not be
split into commits that compile:

- the **device runtime** (context, residency, registry, SYCL backend, USM and CUDA-VMM resources) is one
  include cycle; any split leaves a commit that does not build. 4 planned → 1.
- the **dispatch seam** splits cleanly in two — framework tiers, then the expert hatch — but not into five,
  because all tiers share the mirror, canary and stale-diagnostic helpers. 5 planned → 2.
- **diagnostics** could not lead the series as D5 intended: `DeviceLog.hpp` alone is standard-library-only,
  but `DeviceLoggerBackend.hpp` needs `DeviceContext.hpp` and `qa_DeviceLog.cpp` needs the backend, so the
  smallest _testable_ unit sits after the runtime. Landed at 11.

Phase 5 (SubGraph-owns-dispatch, automatic transfer insertion, the vertical test, host-boundary staging)
is NEW work and lands on top of these 21, per D7.

### 58.6 SLOC audit — the concern was inverted, and two things were rationalised

Measured per commit, splitting **production** from **test/bench** lines, because a reviewer's burden is the
former and test files inflate the raw count without inflating the reading.

**The branch is not "a thousand paper cuts". The opposite:** before rationalising, four commits carried
**65 %** of all production code, and the largest single commit was **1,255 production lines across 18
files**. Fragmentation was never the problem; concentration was.

**Two changes made, net zero commits (21 → 21):**

1. **Split the device runtime** (was 1,255 prod / 18 files) along its real seam — the abstraction, then the
   one implementation of it:
   - `feat(device): the contract a compute backend has to satisfy` — 464 prod (context, residency,
     registry, CPU fallback; no SYCL anywhere)
   - `feat(device): the SYCL backend that implements it` — 791 prod (SYCL context, runtime, USM, CUDA VMM,
     parallelFor)

   This required moving two `parallelFor` cases out of `qa_DeviceContext` into the second commit, since
   they need the SYCL helpers; the first commit's test then exercises only the contract, which is what it
   should have done anyway.

2. **Merged the build and CI commits** (2 and 10 prod lines) into one — they were the same concern stated
   twice, and neither justified a reviewer's context switch.

**Result: largest production commit 1,255 → 791; median ≈ 134 prod lines.**

**Deliberately left alone:** commits 1 and 2 are **1 production line each**. They look like paper cuts and
are not — both are pre-existing `main` bugs (the double-mapped munmap leak and the moved-block resource),
each with its own regression test, deliberately first and self-contained **so a maintainer can cherry-pick
them to `main` without taking any of the device work**. Merging them into anything would destroy that.

Everything else sits between 53 and 634 production lines with a clear single subject, which is the size a
functional review can actually hold.

### 58.7 REGROUPED to 10 commits — the narrative, not the taxonomy

21 commits hid the integration strategy: a reviewer could not see _which few abstractions core needed_ for a
non-CPU domain. Regrouped so the series reads as an argument rather than a changelog. Tree still identical
to `3c3fd663`.

| #   | commit                                                                          | prod+ | test+ |
| --- | ------------------------------------------------------------------------------- | ----- | ----- |
| 1   | `fix(core): release both halves of a double-mapped buffer`                      | 1     | 32    |
| 2   | `fix(core): keep a moved block's memory resource`                               | 1     | 19    |
| 3   | `feat(core): express memory and execution that are not the host's`              | 319   | 290   |
| 4   | `feat(device): a device runtime behind one backend-neutral contract`            | 1255  | 626   |
| 5   | `feat(core): dispatch a block to a device without changing how it is written`   | 934   | 1196  |
| 6   | `feat(device): run a group of blocks as one compute domain`                     | 387   | 577   |
| 7   | `feat(device): diagnostics from inside a kernel, rendered on the host`          | 553   | 325   |
| 8   | `feat(basic): move samples across the host/device boundary explicitly`          | 116   | 78    |
| 9   | `feat(fourier): one FFT that runs on the host or a device`                      | 841   | 3278  |
| 10  | `build,ci,docs(device): compile the stack only when asked, and prove CI ran it` | 138   | 0     |

**The narrative a reviewer now sees:** two pre-existing `main` bugs, cherry-pickable on their own · **the
four core abstractions** (compute-domain grammar, memory-resource capabilities, edge resource precedence,
device-only stream buffer) — none of which mention a device · a backend contract with SYCL as its first
implementation · the dispatch seam · groups as domains · diagnostics · explicit boundary transfers · the
FFT that uses all of it · build/CI/docs.

**Applied from the maintainer's guidance:**

- **tests live with their feature** — the three pure-test/bench commits are gone; `qa_DispatchGroup` now
  ships with the SubGraph it justifies, the acpp device tests with the seam, the benchmarks with the FFT.
- **fixes stay disjunct** — commits 1 and 2 are 1 production line each and stay that way, so a maintainer
  can take them to `main` without any device work.
- **A-B-A avoided where cheap** — `ExecutionStrategy` is now written once (was split across two commits by
  tier); `DeviceContext` once (was contract + implementation).
- **foundational work grouped** — commit 3 is the whole "core learns non-host memory" step, previously four
  separate commits.

`Block.hpp` is still touched twice (commit 3 for the block-tier resource override, commit 5 for the seam) —
two genuinely different concerns, and merging them would produce a 1,250-line commit mixing allocation
policy with dispatch.

Earlier versions kept as `rebuild-v1` (21, unsplit runtime) and `rebuild-v2` (21, split runtime).

### 58.8 Two measurement traps hit in one session — same root cause

Both are "trusted a result taken while the thing under test was changing". Worth naming because neither
announced itself as an error.

1. **`ctest` passed on stale binaries after a failed build.** During the regrouping a commit's build failed
   (`qa_DeviceDomain` missing `DeviceExpectation.hpp`) but the `ctest` in the same shell line ran the
   _previous_ binaries and reported 100 % pass, so the commit landed broken. Fixed by amending — and every
   later step now gates the test run _and_ the commit on the build's exit code (`[ $BUILD -eq 0 ] && …`)
   rather than running them unconditionally.
2. **A background full build raced the history rewrite.** A clang20 build+ctest launched before the 21→10
   regrouping kept running while every `git checkout <target> -- …` moved files underneath it. It reported
   3 build errors and 13 test failures — all uninterpretable, because it compiled a mixture of intermediate
   trees. Discarded and re-run against the stable tree.

**Rule: never launch a long build in the background and then modify the working tree. Either wait, or work
somewhere the build cannot see.** The failure mode is not a crash — it is a plausible-looking failure list
that costs an hour to chase.

## 59. PHASE 5 PLAN (new work, lands on the 10 rebuilt commits)

> **DONE — see §60-§64 for what each piece became.** The vertical test, transfer insertion, the dead-API drop
> and D8 all landed; the one-host-sync-per-group item was built, reviewed and reverted (§63.1).

Per D7 this is development, not restructuring. Four pieces, in dependency order:

| #   | piece                            | why                                                                                                                                                                                                                                  |
| --- | -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 1   | **SubGraph owns dispatch** (D1)  | the scheduler resolves residency once and calls members' processing functions with device buffers, instead of each member re-resolving `compute_domain` per call. Reintroduces `linearChains()` — this time with a consumer (§58.4). |
| 2   | **Automatic transfer insertion** | `makeSubGraph` inserts `HostToDevice`/`DeviceToHost` at the declared boundary. This is what D3 makes possible: membership declares the boundary, so nothing has to be inferred (§56.2).                                              |
| 3   | **The vertical test**            | host → device dispatch → device chain → device→host, on SYCL, plus the corner cases: a member that is not relocatable, a chain broken by a host member, a cycle, two device domains.                                                 |
| 4   | **Host-boundary staging** (D8)   | pinned USM + real `queue.memcpy` in the transfer blocks — the 38 → 0.75 ns/sample fix (§53.4), the branch's only remaining transport lever now that interior residency is handled by the edge PMR.                                   |

Also outstanding, unscheduled: the `Domain` → `SubGraph` rename (§50.7 dispositions; `compute_domain`
stays), and `gr::SubGraph` moving to `core/include/gnuradio-4.0/SubGraph.hpp` per the maintainer's decision.

**Note the rename is now cheap** — §58.4 deleted the chain/epoch API, so the symbols left to rename are just
`Domain`, `DomainWrapper`, `makeDomain`, `DomainHandle` and three file names.

### 58.9 Cross-compiler verification — CLOSED, and no ABI problem

Every failure in the full clang20 run resolved to one of three causes; **none to this branch's changes.**

| test                        | verdict                                                                                                                                                                                                                                                            |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `qa_plugins_test`           | stale binary — **passes** after rebuild                                                                                                                                                                                                                            |
| `qa_grc`                    | stale binary — **passes** after rebuild                                                                                                                                                                                                                            |
| `qa_plugin_schedulers_test` | never rebuilt in any pass — **passes** once built                                                                                                                                                                                                                  |
| `qa_SubGraphAssets`         | **pre-existing clang20 `-Werror` failure on `origin/main`**: `schedulerThread.get()` ignores a `[[nodiscard]]` at `:485`/`:533`; the file is byte-identical to main's. Its SEGFAULT was a stale binary.                                                            |
| 5 build errors              | **pre-existing `main` defect** — cpr's `throw` reaching the deliberately no-exceptions `qa_Embedded` TU via `Graph_yaml_importer.hpp → PluginLoader.hpp → FileIo.hpp → cpr.h`. Introduced by main's own tip commit `7a8e2e54`. Confirmed on gcc15 **and** clang20. |

**The question that mattered: does the +64 B `Block<T>` growth break plugin ABI?** All four plugin
segfaults were stale binaries; every one passes when actually rebuilt. **The one-layout ODR decision holds.**

**One real defect of ours, found and fixed:** `bm_fft_stream_graph.cpp` captured a `constexpr` in two
lambdas — clang20 `-Wunused-lambda-capture`, invisible to GCC. Folded into the FFT commit by non-interactive
autosquash (`GIT_SEQUENCE_EDITOR=true git rebase -i --autosquash --committer-date-is-author-date`), so the
history carries no commit repairing code introduced one commit earlier.

**Two upstream reports owed to `main`:** the cpr/no-exceptions break and the `qa_SubGraphAssets` nodiscard.
Both make `origin/main` un-buildable with `-Werror` on a full build.

**Method rule earned three times over:** a test failure following a non-zero build exit is _unexamined_, not
a finding. Stale binaries and a background build racing a `git checkout` each produced a plausible failure
list that cost real time.

### 58.10 Rename folded — `Domain` never enters the history

The rename had landed as an 11th commit _after_ the one that introduced the old name — the A-B churn the
maintainer had objected to. Folded properly by rebuilding commits 6-10 from the already-renamed tree, so
the vehicle is called `gr::SubGraph` from its first appearance.

**Verified mechanically**, not by inspection: every commit's added lines were grepped for
`makeDomain` / `DomainWrapper` / `DomainHandle` / `device::Domain` / `DeviceDomain.hpp`. **Zero hits across
the whole series.** That check caught one leftover the earlier word-boundary sed had missed — a local
`makeDomain()` helper inside `qa_DispatchGroup` (its own test-local group type), now `makeSyncGroup()`
returning a `SyncGroup`, so the framework-contract test stays legible beside the vehicle without sharing
its name.

**Final: 10 commits, tree identical to the renamed target, full gcc15 build with zero non-cpr errors, 19/19
affected tests pass.**

| #   | commit                                                                          | date       |
| --- | ------------------------------------------------------------------------------- | ---------- |
| 1   | `fix(core): release both halves of a double-mapped buffer`                      | 2026-08-07 |
| 2   | `fix(core): keep a moved block's memory resource`                               | 2026-07-09 |
| 3   | `feat(core): express memory and execution that are not the host's`              | 2026-07-10 |
| 4   | `feat(device): a device runtime behind one backend-neutral contract`            | 2026-07-09 |
| 5   | `feat(core): dispatch a block to a device without changing how it is written`   | 2026-07-09 |
| 6   | `feat(core): run a group of blocks as one scheduling sub-graph`                 | 2026-08-21 |
| 7   | `feat(device): diagnostics from inside a kernel, rendered on the host`          | 2026-07-09 |
| 8   | `feat(basic): move samples across the host/device boundary explicitly`          | 2026-08-08 |
| 9   | `feat(fourier): one FFT that runs on the host or a device`                      | 2026-08-13 |
| 10  | `build,ci,docs(device): compile the stack only when asked, and prove CI ran it` | 2026-08-09 |

~~`qa_FftCrossover` is flaky under parallel ctest~~ — **MOOT 2026-08-25: the test was dropped.** It measured
where the device transform overtakes the host one, so a loaded machine moved the answer; that made it a
debugging instrument rather than an assertion, and a `RUN_SERIAL` property would have papered over it. The
crossover _finding_ — the GPU wins the FFT only at N >= 4096 **and** batch >= 16 — never came from this test
anyway: it comes from the benchmark sweep in §53, which is intact. Nothing that a decision rests on was lost.

Branches kept: `rebuild-v1` (21, unsplit runtime), `rebuild-v2` (21, split runtime),
`rebuild-v3-with-rename` (11, rename as a separate commit). Current work is on `rebuild-v3`.

## 60. PHASE 5 — the vertical test (2026-08-24, commit 11)

`core/test/qa_SubGraphVertical.cpp`: `source -> SubGraph[ HostToDevice -> member -> member -> DeviceToHost ]
-> sink`, run end to end **on a real SYCL device** (`build-acpp`, `ACPP_VISIBILITY_MASK=omp`, `host:sycl`)
and asserted bit-identical against the same group on the host. **5 tests / 49 asserts pass on acpp; passes
on gcc15 with the device legs skipping.**

The transfers sit INSIDE the group on purpose — membership declares the boundary, so the crossing points
belong to the group that owns it. **That makes this file the specification for the automatic-insertion
helper, not just a test of it.**

### 60.1 The failure it hit was the test harness, not the branch — G10 again

First acpp run: **zero samples through the device leg**, with

```
[AdaptiveCpp Error] jit::compile: Could not obtain HCF object
omp_queue: Code object construction failed
```

Diagnosis, by elimination rather than guesswork:

1. `qa_DeviceAutoParallel` and `qa_DeviceBlockStyles` **pass** on the same build, same env, and
   `qa_DeviceBlockStyles` genuinely dispatches to the device — so not the toolchain.
2. Suspected `gr::testing::Copy` also living in the block library (two definitions, `--icf=safe` folding).
   **Refuted** — a test-local unregistered block failed identically. The local block was kept anyway, so
   the question cannot come back.
3. Real cause, already documented as **gotcha G10** in `qa_DeviceBlockStyles.cpp:149`: _"AdaptiveCpp aborts
   if a kernel is launched while Boost.UT is running suites from ~runner (static destruction)"_. Every
   working device test drives from `main()`; mine used a global `const suite<>`, which runs at static
   destruction. Restructured to `main()` → all green.

**Cost of not reading the existing device tests' structure first: one wrong hypothesis and two build cycles.**
G10 was written down precisely for this and it is worth re-reading before adding any device test.

### 60.2 Layering constraint for the insertion helper (next piece)

`SubGraph.hpp` is **core**; `TransferBlocks.hpp` is **blocks/basic**. Core cannot depend on blocks, so
`makeSubGraph` cannot instantiate `HostToDevice<T>` itself. Automatic insertion must be a `blocks/basic`
helper, e.g. `gr::basic::makeDeviceSubGraph<float>(gr::Graph&&, "gpu:sycl")`.

Requiring the caller to name the sample type is registry-free and consistent with D3. Instantiating by type
name through the block registry would make the feature unavailable in registry-OFF builds — the same trap
that ruled out routing sub-graph formation through `groupBlocks` (§56.2).

## 61. PHASE 5 — automatic transfer insertion (2026-08-24, commit 12)

`gr::basic::makeDeviceSubGraph<T>(gr::Graph&&)` — the workflow the maintainer described: build an ordinary
graph, hand it over, get back a group whose boundary already carries its transfers.

**7 tests / 71 asserts pass on a real SYCL device**; 5/5 on gcc15 with the device legs skipping.

### 61.1 What it does

Inserts `HostToDevice<T>` in front of every port no interior edge claims, `DeviceToHost<T>` after every
unclaimed output, and exports the **transfers'** outer ports rather than the members' inner ones. What a
caller connects is unchanged; what happens behind that port becomes a copy.

### 61.2 Three decisions worth keeping

- **`boundaryPorts()` lifted out of `makeSubGraph()`** so both share one definition of "which ports would be
  exported". A helper that re-derived the scan would silently drift from what the group actually exports.
  `refuseTwoDeviceDomains()` moved out with it. All existing suites still green.
- **It does NOT set `compute_domain` on the members.** That was the first design, abandoned on reading
  §50.7: `settings().set()` through a `BlockModel` never reaches the concrete field, which is exactly how
  the `disconnect_on_done` override sat there doing nothing for weeks. Members carry their own domain from
  `emplaceBlock`; the helper only says where data crosses. Also more faithful to D3.
- **`T` is named, not deduced.** A graph is type-erased by the time it reaches the helper; resolving each
  port's type would mean the block registry, which would make the feature vanish in registry-OFF builds --
  the same trap that ruled out routing group formation through `groupBlocks` (§56.2). **Known limit: a
  mixed-type boundary is not expressible yet.** Worth stating before it surprises someone.

### 61.3 Layering, confirmed by construction

`SubGraph.hpp` is core and must not know about blocks; `TransferBlocks.hpp` is `blocks/basic`. The helper
therefore lives in `blocks/basic`, and core stays unaware that transfer blocks exist. That is the right
split: the _vehicle_ is a core concept, the _transfers_ are ordinary blocks.

**Phase 5 remaining:** SubGraph-owns-dispatch (D1), host-boundary staging (D8).

## 62. PHASE 5 — one host sync per group (D1), and what D8 is blocked on (2026-08-24, REVERTED)

> **The commit this section describes was reverted (§63.1) and is in no history.** "commit 13" here and in §63
> means that reverted commit, not today's commit 13. Numbering below §63 shifted by one when the
> `qa_FftCrossover` `RUN_SERIAL` commit was dropped on 2026-08-25; the section headings have been corrected, the
> body text of §62-§63 has not.

### 62.1 D1 landed as batching, not as a BlockModel vtable change

The premise that made it small: **every SYCL queue this runtime creates is in-order** —
`SyclRuntime.hpp:46` (default queue) and `:150` (each enumerated queue). A member's kernel therefore already
runs after its predecessor's, so the per-submission wait buys ordering nothing and costs a host round trip.
A chain of N device members paid N; a `SubGraph` now opens a batch, drives its members, and pays **one**.

`SubGraph` regains a `_context` — the §58.4 note said phase 5 would reintroduce it _with a reader_, and this
is that reader.

**Two hazards, both found by running it rather than reasoning about it:**

1. **A mid-chain CPU fallback reads results early.** `dispatchCpuFallback` now drains the batch first. It is
   the slow path already, so correctness is free there. Call sites that have a context pass it; the two that
   fire _before_ resolution legitimately cannot.
2. **Freeing USM under a running kernel.** `dispatch` releases its per-call scratch the moment it has
   submitted; with the wait removed that is a free under a live kernel. Observed as
   **`free(): corrupted unsorted chunks`**, not predicted. `deallocate()` now defers to a pending list while
   batching and frees after `endBatch()` has waited.

Opt-in by design: `beginBatch()` defaults to a no-op and `endBatch()` to a plain `wait()`, so a backend that
cannot defer keeps its per-submission wait and nothing about the contract changes.

**Verified:** `qa_SubGraphVertical` 71/7, `qa_SubGraph` 77/10, `qa_DispatchGroup` 24/2,
`qa_DeviceBlockStyles` 53/6, `qa_DeviceAutoParallel` 26/4 — all on acpp/SYCL, no corruption; 5/5 on gcc15.

### 62.2 D8 is NOT done, and deliberately so

Half of it already existed: `HostToDevice`/`DeviceToHost` have used bulk `queue.memcpy` all along. The
missing half is **pinned host staging** — `UsmMemoryResource` allocates `aligned_alloc_shared` for
everything, and shared-USM read-back is precisely what measured **38 ns/sample**.

**Not attempted, because it cannot be honestly validated right now.** `nvidia-smi` shows 6346/8192 MiB held
by another job and the clock unpinned; the measurement protocol requires an idle card _and_ a pinned clock,
and pinning needs a real TTY the maintainer owns. Shipping a change that claims ~50× without measuring it is
the exact failure this branch has recorded before. **Owed: implement pinned staging, measure on an idle
pinned card, and only then claim the number.**

### 62.3 A third pre-existing `main` defect

`blocks/sdr/RTL2832Source.hpp` (+ its example and test) ignores `[[nodiscard]]` returns in ~40 places,
failing acpp/clang `-Werror` where gcc does not warn. `git diff origin/main HEAD -- blocks/sdr/` is empty —
untouched by this branch. **Three upstream reports now owed to `main`:** cpr in the no-exceptions TU,
`qa_SubGraphAssets`'s nodiscard, and this.

## 63. ADVERSARIAL REVIEW OF PHASE 5 — one commit reverted, two real bugs fixed (2026-08-24)

An independent review attacked the three phase-5 commits. It was largely right. Outcome: **commit 13
(batching) reverted**, **commit 12's two defects fixed**, several findings still open.

### 63.1 REVERTED — `perf(core): one host synchronisation per sub-graph`

Three findings, each sufficient on its own, all verified independently before acting:

1. **The headline claim was FALSE.** `DeviceContextSycl::pollDeviceError()` does an unconditional
   `queue->wait()` (`:185`) and every device dispatch path ends by calling it (`ExecutionStrategy.hpp:190,
:363, :439`). A chain of N members therefore still paid N host synchronisations. The batch only collapsed
   the _intra-dispatch_ waits (H2D + kernel + D2H, ~3→1) — a real but much smaller and **unmeasured** win.
2. **A third hazard I had missed.** `dispatchDeviceBulk` reads the kernel's status word `*statusPtr` from
   shared USM on the host (`:355`) **before** the `pollDeviceError()` sync (`:363`). Under a batch the
   kernel may still be running: a device `processBulk` returning ERROR would be reported as the pre-seeded
   OK, and it is a concurrent host/device access to managed memory. The vertical test could not see it —
   its members use `processOne`, which has no status word.
3. **Re-entrancy UB.** `_batching` was a plain `bool` on the **per-domain registry singleton**, shared by
   every group _and_ every ungrouped device block on that domain. Under `scheduler::Simple<multiThreaded>`
   two of them interleave: concurrent `push_back` on `_pendingFrees` is UB, and one group's `endBatch()`
   clears `_batching` while another is mid-loop — reintroducing exactly the "free(): corrupted unsorted
   chunks" the commit had fixed.

The guard was in the wrong place: the hazard class is _any host consumer of a batched producer_, not "CPU
fallback". A correct version needs depth-counted, owner-scoped batching and a sync before every host read of
device memory. **Kept: the `RUN_SERIAL` fix, re-committed alone.**

### 63.2 FIXED — `feat(basic): insert a group's transfers`

- **The transfers never transferred.** No `compute_domain` was set on them, so they took the host
  `std::ranges::copy` path: two extra host copies and no transfer. My stated reason (settings-through-
  BlockModel does not apply) was wrong — the helper _constructs_ the blocks and can pass the domain in the
  `emplaceBlock` init map, which the test already does for members. Now takes `deviceDomain` explicitly.
- **≥2 boundary ports per side was refused.** Two unnamed transfers of one type export the same port name.
  Now each is named.

### 63.3 A hang I misdiagnosed — CORRECTED 2026-08-25

Naming the transfers after the member's **unique** name made the whole suite hang, and I concluded the cause
was `::`/`#` corrupting the `<block name>:<port>` export name, recommending that `boundaryName()` reject a
colon.

**That conclusion was wrong, and implementing it proved it wrong.** A block's name defaults to its _type_
name, so every namespaced block already carries `::` — the shipped and passing export names include
`gr::basic::HostToDevice<float32>:in`. A guard rejecting ':' refuses essentially every block; applied, it
broke `qa_SubGraph` and `qa_SubGraphVertical` outright. Reverted.

**So the real cause of that hang is still unknown.** What is established: naming the transfers by a plain
index works, naming them by unique name hangs, and `::` in a name is demonstrably fine on its own. The
difference is more likely the `#` or the resulting name length/uniqueness interaction, and it deserves a
proper bisect rather than another guess. **Do not re-derive the ':' theory — it has been tested and refuted.**

### 63.4 STILL OPEN from the review

> **RE-VERIFIED 2026-08-25 — three of these four are closed.** The `refuseTwoDeviceDomains` bypass was fixed
> (`startDispatch()` re-checks it, `SubGraph.hpp:71`); the batching regression tests stayed moot, batching having
> been reverted; the transfer counters were deleted as dead API in commit 13. Only the mixed-`T` boundary remains,
> and §64 records it as a documented limitation rather than an open defect.

- `refuseTwoDeviceDomains` guards only the two factory functions; `SubGraphWrapper`, `setGraph()` and
  `graph()` all bypass it, so the single-domain invariant is not enforced on the type. Cheap fix: re-check
  in `startDispatch()`.
- `makeDeviceSubGraph` uses one `T` for all boundaries — a mixed-type boundary gets a wrong-typed transfer,
  untested.
- No regression test for either batching hazard (moot while reverted, needed if it returns).
- `hostTransferCount()`/`hostDownloadCount()` exist precisely to assert crossings and no phase-5 test uses
  them — the insertion helper's effect is therefore asserted structurally, not by counting transfers.

## 64. DEAD API DROPPED (2026-08-25, commit 13)

Per the maintainer's rule — _GLSL/WebGPU-only and unlinked to the CPU/SYCL chain gets dropped; anything kept
needs its merit documented_:

| dropped                                                                                                              | why                                                                                                                            |
| -------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| `hostTransferCount()` / `hostUploadCount()` / `hostDownloadCount()` and their `_hostUploads`/`_hostDownloads` fields | built to make the GLSL elision assertable ("34 bound vs 66 unbound"); **zero readers** anywhere after the shader backends left |
| `copyDeviceToDevice()` (pure virtual + both implementations)                                                         | moved data between two opaque SSBOs; **zero callers** on the CPU or SYCL path                                                  |

Both were verified to have no remaining consumers before removal, then re-verified by building gcc15 and
acpp clean and running the six affected suites.

**Still-open review item, now with a documented reason to keep:** `makeDeviceSubGraph` uses one `T` for every
boundary. An attempt to _check_ this (compare each port's `metaInfo.data_type` against `type_name<T>()`)
was written and abandoned — it segfaulted, and the cause was not chased to ground. The limitation therefore
stands and is documented at the call site rather than half-guarded: **a group with mixed boundary types
needs one `makeDeviceSubGraph` call per type.**

## 65. WHY THE DEVICE LOGGER EXISTS — settled empirically (2026-08-25)

Asked whether the generic logger would suffice if it were merely handed a device USM-PMR resource. **It would
not, and memory placement was never the obstacle.** Two things in the generic emit path cannot execute on a
device: `Backend::publish()` is virtual (`Logger.hpp:52`, and SYCL forbids function pointers in device code —
the same reason `parallelFor` is not a virtual on `DeviceContext`), and `publishFormatted()` calls
`std::format`. `LogRecord` itself _is_ a device-safe POD, but its `text[]` holds **already-formatted** text,
which is precisely what a kernel cannot produce.

**Tested rather than argued.** `std::format_to_n` into a pre-allocated buffer inside a `single_task`:

| target                                | result                                      |
| ------------------------------------- | ------------------------------------------- |
| `--acpp-targets=omp`                  | compiles and runs — `value=42 ratio=1.50`   |
| `--acpp-targets=generic` (SSCP → PTX) | compiles; fails at code-object construction |

Peeling the SSCP failure gave three layers: `~runtime_error` via `format_error` (**removable** with
`-fno-exceptions`), `memchr` unresolved (**supplyable** in ~8 lines via an `__asm__` label), and finally
**`std::locale::~locale`** — libstdc++ materialises a `std::locale` on the format path so the `L` specifier
can work, and that is a reference-counted, virtually-dispatched, globally-registered facet table. Not
strippable, and not something a device stub can honestly fake.

**So the buffer-args-and-render-on-the-host design is not a workaround for a solvable problem** — it is the
same conclusion CUDA's device `printf` reached, and it works today on every backend with full formatting
available host-side. Reported upstream separately; **closed as far as this branch is concerned.**

There is exactly ONE logger: `LogRecord`, `Backend`, the drain and the formatting policy are shared, and
`DeviceLoggerBackend` _is_ a `gr::log::Backend`. What the device adds is a front end for kernels, which
converges back onto the generic `LogRecord` at drain time — not a parallel implementation that can drift.

## 66. D8 MEASURED — the recorded finding reproduces, but the headline was the wrong half (2026-08-25)

Run on an idle RTX 3070, clock pinned 1500 MHz, best-of-two runs (JIT cache warm on the second).
Probe: `scratchpad/bm_readback.cpp`. Each iteration does what a device block actually does — **a kernel
writes the buffer, then the host reads every element.**

| samples | A shared USM | B device + pageable staging | C device + pinned staging | A/C       |
| ------- | ------------ | --------------------------- | ------------------------- | --------- |
| 16 Ki   | 37.21        | 2.06                        | 1.53                      | 24.3×     |
| 256 Ki  | 17.06        | 1.26                        | 1.02                      | 16.7×     |
| 1 Mi    | 17.32        | 1.16                        | 1.00                      | 17.3×     |
| 16 Mi   | 17.18        | 1.15                        | 0.98                      | **17.6×** |

ns/sample, median of 15 after 3 warm-up rounds.

**The recorded numbers reproduce**: 17.2 ns/sample for shared-USM read-back (recorded: 38) and 0.98 for the
staged path (recorded: 0.75) — same regime, and the ~17× ratio is the same story.

### 66.1 The correction that matters

**The win is staging, not pinning.** Shared USM → any staging is **15×**; pageable → pinned is only **1.17×**.
D8 was framed as "pinned USM + real `queue.memcpy`", which put the emphasis on the 17 % rather than the 1500 %.
The actual mechanism is: **let the kernel write device-only memory and bring it back with one bulk copy,
instead of letting the host fault shared-USM pages back one at a time.**

`queue.memcpy` is already in the transfer blocks. What is missing is that `UsmMemoryResource` hands out
`aligned_alloc_shared` for everything, so the buffer a kernel writes is shared USM and the host read faults it
back page by page — which is precisely what column A measures.

### 66.2 The first probe was wrong, and physics caught it, not the numbers

`bm_host_boundary.cpp` copied buffer-to-buffer and never read host-side, so the managed pages stayed
device-resident and shared USM clocked **200 GB/s** — impossible over a ~25 GB/s PCIe Gen4 x16 link. It was a
device-to-device copy in a host-boundary costume, and it would have been reported as "shared USM is 7× faster
than pinned, D8 refuted" had the bandwidth not been checked against the hardware ceiling.

**Rule: convert a throughput result to GB/s and compare it against the link before believing it.** A
ns/sample figure carries no units that contradict a wrong setup; a bandwidth does.

### 66.3 What D8 should now be

Not "add pinned staging to the transfer blocks". Rather: **a device edge that only kernels touch should be
device-only memory, and the host boundary should be one bulk copy into host staging** — with pinning a
1.17× refinement on top, worth taking but not the point.

That is close to what `Access::DeviceOnly` + `CudaVmmMemoryResource` already do for _interior_ edges
(§30.1 measured it). The gap is the **boundary**: the buffer at the group's edge is still shared USM.

**Still to do, and now with a measured target:** make the transfer blocks' device-side buffer device-only and
their host-side buffer pinned staging, then re-run this probe end-to-end through `qa_SubGraphVertical`'s
topology rather than in isolation.

## 67. D8 IMPLEMENTED — and the end-to-end benchmark redefined what it is worth (2026-08-25, commit 16)

### 67.1 What shipped

A device edge whose endpoints differ — one side computing on the device, the other on the host — now resolves
to **pinned host USM** instead of shared USM. `UsmMemoryResource` gained a `UsmKind{shared, hostPinned}`,
`SyclRuntime` a `pinnedHostResourceFor(queue)`, and `Graph::applyEdgeConnection` sets `Access::HostOnly` for
such an edge (it previously fell through to `Shared`).

### 67.2 The isolated measurement (RTX 3070, idle, 1500 MHz pinned)

Kernel writes the buffer, host reads every element — 15 iterations, 3 warm-up:

| samples | shared USM | device + pageable | device + pinned | ratio     |
| ------- | ---------- | ----------------- | --------------- | --------- |
| 1 Mi    | 17.32      | 1.16              | 1.00            | 17.3x     |
| 16 Mi   | 17.18      | 1.15              | 0.98            | **17.6x** |

Reproduces the recorded 38 / 0.75 figures (same regime). **The win is staging, not pinning**: shared → any
staging is 15x, pageable → pinned is only 1.17x.

### 67.3 The end-to-end result: no change, and the reason matters more than the number

An A/B of the full `source → SubGraph[h2d → member → member → d2h] → sink` topology showed **nothing outside
noise** (+3.6 %, −3.9 %, signs both ways). Arithmetic said it should have doubled: `gpu:sycl` at 2 Mi runs
54 MS/s ≈ 18 ns/sample, and isolated shared-USM read-back is 17.

Rather than conclude either way, the ring was **classified at run time** with `sycl::get_pointer_type`:

| topology                             | domain      | boundary ring                                         |
| ------------------------------------ | ----------- | ----------------------------------------------------- |
| flat (device block → host sink)      | `gpu:sycl`  | **host (pinned)** — the change works                  |
| grouped (SubGraph + transfer blocks) | `gpu:sycl`  | unknown (plain heap) — change inert                   |
| either                               | `host:sycl` | unknown — `kind=="host"` skips the USM tier by design |

**So the null result was correct and the change is correct; they are about different topologies.**

- A **group's exported boundary port does not carry its member's `compute_domain` outward**, so the parent
  edge is an ordinary host edge and never sees the USM tier at all.
- A `HostToDevice`/`DeviceToHost` pair already moves that data with one bulk `queue.memcpy`, which is the
  _staging_ path — so the group topology was already collecting the 15x and only the 1.17x remained, well
  inside the benchmark's noise.

**The explicit-transfer design was already avoiding the 17x. This commit fixes the case that does not use
it** — a flat graph where a host block reads a device block's output directly.

### 67.4 Method notes worth keeping

- **The first probe was wrong and looked right.** It copied buffer-to-buffer without a host read, so managed
  pages stayed device-resident and shared USM clocked **200 GB/s** — impossible over a ~25 GB/s PCIe Gen4
  x16 link. It would have been reported as "D8 refuted". **Convert a throughput number to GB/s and check it
  against the link before believing it**; ns/sample carries no units that can contradict a broken setup.
- **A null A/B is not an answer.** Classifying the artefact (what memory is this, actually?) turned "no
  effect" into "no effect _here_, full effect _there_", which is a different and correct conclusion.
- Backticks inside a double-quoted shell commit message are command substitution: `"…now `host (pinned)`…"`
  silently committed "now ". Use a heredoc for anything with backticks.

**Diagnostic kept out of the branch** (`scratchpad/bm_readback.cpp`, `bm_boundary_graph_final.cpp`) — it
answered its question and would otherwise be a benchmark target with one historical consumer.

## 68. DEVICE-TO-DEVICE — MOSTLY ALREADY WORKS; the residual is one narrow limitation (corrected 2026-08-26)

> **CORRECTION.** This section previously claimed a device block could not fan out to several device consumers
> without a host round trip. **That is false, and was measured false** — the claim was inherited from the chain
> machinery removed under D5, whose 1:1 restriction was about writing IN PLACE into one shared buffer, not about
> capability. Residency is per EDGE now, and a port's output ring is naturally multi-reader.

**Probe (`scratchpad/fanout_probe.cpp`, RTX 3070, `gpu:sycl`)** — `source -> head -> {armA, armB} -> sinks`, all
three Gain blocks on one device domain, reporting `gr::isDeviceOnly(edge._dataResource)` per edge:

| edge             | device-only | resource                        |
| ---------------- | ----------- | ------------------------------- |
| source -> head   | no          | 0x…c40d0                        |
| **head -> armA** | **YES**     | 0x…d350d0                       |
| **head -> armB** | **YES**     | **0x…d350d0 — the same buffer** |
| armA -> sinkA    | no          | 0x…c40d0                        |
| armB -> sinkB    | no          | 0x…c40d0                        |

One device ring, two device readers, no host round trip. USM serves device-to-device exactly as it serves
host-to-device (maintainer, 2026-08-26). The boundary edges are correctly host-resident.

### 68.0 What is actually left

1. **Group-to-group chaining (the real limitation).** A `SubGraph`'s exported boundary port does not carry its
   member's `compute_domain` outward, so the parent edge is an ordinary host edge (§67.3, classified at run time
   with `sycl::get_pointer_type`). Within a group and in a flat graph, device-to-device is the normal case; only
   _between_ groups does it fall back to the host. **Not a blocker** — a group boundary is usually meant to be a
   host boundary, and anything needing device-to-device across it can be one larger group today.
2. **In-place elision across a chain — a performance item, not a capability one.** Each device edge owns its own
   ring, so a chain of N device blocks holds N device rings rather than one shared buffer. The removed machinery
   collapsed those; nothing is incorrect without it, only less frugal with device memory and bandwidth.

**Mechanisms if anyone revisits (2), deliberately unranked — the choice follows a measurement:** a device-side
circular buffer (the ring contract the host edges already use, in device memory); shared A/B data chunks handed
between blocks on one device (aliasing sidestepped: a writer and a reader never hold the same chunk); copy-on-fan-out
via a reinstated `copyDeviceToDevice` (cheapest, the baseline the others must beat); read-only or versioned buffers.

**The workloads that would justify it are below.** Everything in §68.1-§68.4 was written under the false premise
and is kept for the numbers, not the framing.

### 68.1 The two workloads

**FFT-based FIR (fast convolution).** Overlap-save is `FFT -> multiply -> iFFT`: three blocks, two interior
edges, and nothing in the middle the host has any reason to see. §53.5 already identified this as the
litmus case — the useful GPU envelope is N >= 4096 with batch >= 16, which is a real DSP regime
(channelisers, matched filters, pulse compression), and §53.4 showed the transfer budget is what decides
whether the 5x kernel advantage survives at all: a naive host round trip turns it into a 10x LOSS.

**Channeliser.** One input stream, fanned out internally to many data blocks, each further processed, and
only the final results returned to the host. The interior traffic is larger than the boundary traffic by the
channel count — precisely the shape where keeping data resident pays most.

### 68.2 What already works

An edge whose **two endpoints share one device domain** is already `Access::DeviceOnly` and resolves to
device-only memory (`Graph.hpp` sets it; §30.1 measured that an interior edge really is device-resident, in a
**flat** graph, committed as `qa_DeviceResidency`). So the straight `FFT -> multiply -> iFFT` chain, all three
members on one device domain, already keeps its interior on the device today.

### 68.3 What does NOT work, and is the actual gap

1. **Fan-out.** The chain logic that existed was explicitly limited to _maximal 1:1 linear runs_: a buffer two
   members read must never be written in place. A channeliser is fan-out by definition, so the one-to-many
   split is exactly the case no interior-residency scheme here has ever covered. It needs one of the mechanisms
   listed at the head of this section, not an extension of the 1:1 rule.
2. **Device-to-device copy.** `DeviceContext::copyDeviceToDevice()` was deleted in commit 13 — zero callers,
   GLSL-only heritage. A fan-out that cannot alias needs exactly this: one device buffer copied to several
   without a host round trip. **Reinstating it is the cheap part; deciding the aliasing policy is not.**
3. **The group boundary loses domain information** (§67.3): a `SubGraph`'s exported port does not carry its
   member's `compute_domain` outward, so the parent edge is an ordinary host edge. Fine while the boundary is
   _meant_ to be host memory, but it means a group cannot currently be chained to another group
   device-to-device.

### 68.4 What to measure before building any of it

§53 is the standing caution: the GPU only wins for FFT at N >= 4096 **and** batch >= 16, and at batch = 1 it
loses at every size. A channeliser with 64-1024 channels sits below that crossover on FFT size alone, so
"keep it on the device" must be justified by the **interior traffic it removes**, not assumed from the
compute. The instrument for that is a transfer count at the boundary — which was also removed in commit 13
(`hostTransferCount()`), for the same good reason, and would need reinstating alongside.

**Order of work, when it comes:** build the transfer counter first — it is the instrument, and without it every
later claim is unmeasured. Then `copyDeviceToDevice` as the baseline, because it is the cheapest of the four
mechanisms and gives the others something to beat. Only then choose between the device-side ring and shared A/B
chunks, against a real channeliser rather than in the abstract. The earlier framing — _decide the aliasing policy
first_ — was wrong in ordering: the policy follows the measurement, not the other way round.

## 69. THE CROSSOVER HARNESS DROPPED, AND THE BRANCH RE-VERIFIED AT 16 COMMITS (2026-08-25)

`qa_FftCrossover` was a **debugging instrument, not a test**: it searched for the N and batch at which the device
transform overtakes the host one, so its answer moved with machine load (§ the flakiness note above). It was folded
out of commit 9 and the `RUN_SERIAL` commit that existed only to contain its flakiness was dropped with it, taking
the branch from 17 commits to 16. Nothing else changed — `git diff backup/pre-crossover-drop HEAD --stat` is exactly
`qa_FftCrossover.cpp` (−294) and `algorithm/test/CMakeLists.txt` (−6, three lines from each commit).

**Nothing that a decision rests on was lost.** The crossover _numbers_ — the GPU wins the FFT only at N >= 4096 and
batch >= 16 — were never produced by this test, which only ever asserted orderings. They come from the benchmark
sweep in §53 and from `reference_fft_backend_benchmark_table`, both intact, and `bm_FFT_backends` /
`bm_fft_stream_graph` still reproduce them on demand.

**Re-verified after the replay** (acpp, `build-acpp`, 0 errors, `-j4`):

| suite                    | asserts / tests |
| ------------------------ | --------------- |
| `qa_algorithm_fourier`   | 2764 / 44       |
| `qa_SubGraph`            | 87 / 10         |
| `qa_SubGraphVertical`    | 77 / 9          |
| `qa_HostToDevice`        | 20 / 4          |
| `qa_DeviceLog`           | 25 / 7          |
| `qa_DeviceLoggerBackend` | 18 / 4          |

All rc=0. **2991 asserts, 78 tests, no failures.**

**Trap worth keeping:** `pgrep -f "make -C build-acpp"` inside a wait loop **matches the wait loop's own command
line**, so the loop never exits and the build looks like it is still running long after it finished. Anchor the
pattern (`pgrep -f "^make -C ..."`) or match the log instead.

## 76. A DEVICE `InputSpanLike`/`OutputSpanLike` — the concepts ARE the seam (2026-08-26)

**Maintainer's proposal:** keep `processBulk(InputViewLike…)` as the marker for fixed batched work that touches
neither tags nor settings, and _additionally_ enable the classical
`processBulk(gr::InputSpanLike auto&, gr::OutputSpanLike auto&)` on a device by giving the concepts a device-side
implementation — they are concepts, not concrete types, so nothing forces the port spans.

**Correct, and it decomposes into three stages of very different size.** What the concepts actually require
(`Port.hpp:421` and `:445`):

| requirement                                    | device-side answer                                                                                                                                                                                        |
| ---------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `contiguous_range`, `ConstSpanLike`/`SpanLike` | trivial — a pointer and a size                                                                                                                                                                            |
| `isConnected`, `isSync`                        | trivial — two bools carried by value                                                                                                                                                                      |
| `consume(n)` / `publish(n)`                    | a counter in device-visible memory; the host applies it to the REAL span after the kernel                                                                                                                 |
| `rawTags()` → range of `gr::Tag`               | a `std::span<const Tag>`: `Tag` is trivially copyable by static_assert (`Tag.hpp:93`) and a device edge's tag axis is already `Access::Shared`, so the tags are device-dereferenceable **where they lie** |
| `tags()`, `tags(n)`, `consumeTags(n)`          | index-filtered views over the same span plus a counter                                                                                                                                                    |
| `publishTag(property_map&, offset)`            | **the one real blocker** — `property_map` is owning, and a kernel cannot allocate                                                                                                                         |
| `span.tags` must itself be `WriterSpanLike`    | a writer span over a pre-reserved device tag slab                                                                                                                                                         |

### 76.0 IMPLEMENTED — all five gaps, demonstrated end to end (2026-08-26)

`DeviceSpans.hpp` (a device `InputSpanLike`/`OutputSpanLike` pair, compile-verified against the real concepts),
kernel-side `consume`/`publish` recorded in a `DeviceSpanAccounting` and replayed onto the real port spans,
a dispatch tier ordered before the view tier, `copyBackUserState()` for single-work-item state, and a device
`publishTag` that copies a view's blob into a host-pre-reserved slot for the host to replay.

**Two probes decided the design, and both corrected a prior belief.**

1. **`publishTag` already accepted a view.** It is `template<WireMapLike TPropertyMap>`, and `WireMapLike` names
   `ValueMapView` alongside `property_map` (`Tag.hpp:69`); it only ever wanted `tagData.blob()`. The signature
   change we were contemplating had been made long ago — what blocks a kernel is _constructing_ a
   `property_map` literal, not passing one.
2. **A kernel can build a tag payload — in a pre-reserved slot AND in a kernel-local buffer.** Measured on an
   RTX 3070 via `scratchpad/tagbuild.cpp`. `ValueMapView::formatAt` documents itself device-callable; the first
   run said "refused" only because the capacity arithmetic overflowed the slot (`sizeof(Header)=32` plus
   `8 x sizeof(PackedEntry)=48` leaves 96 payload bytes in 512, not the 256 assumed).

**`qa_DeviceSpans` demonstrates it with a discriminator, not by assumption.** Two blocks, identical logic,
differing only in payload form:

| block                     | payload                                 | path taken                    |
| ------------------------- | --------------------------------------- | ----------------------------- |
| `ZeroCrossingTrigger`     | `property_map{...}` (owning, allocates) | canary fires -> host fallback |
| `ZeroCrossingTriggerView` | `formatAt` + `try_emplace` -> view      | runs as a kernel              |

Result: **40 asserts / 3 tests, rc=0, and exactly 2 fallback warnings** — one per SYCL domain, from the owning
form only. Both blocks tag `{32, 64, 96}` on `host`, `host:sycl` and `gpu:sycl`. Matching output alone could not
have proved this (both paths produce identical tags by construction), which is why the warning COUNT is the
assertion.

**A pre-existing defect surfaced on the way.** `dispatchDeviceBulk` asked `isFirstUseOfTheseSettings(block)`
_after_ `deviceMirror()` had already refreshed the epoch that question reads, so its mutation canary — the guard
against "processBulk mutates the block; a device copy would discard those writes" — **had never fired** for a
block owning a shadow. The auto-parallel path had the ordering right, which is what identified the intended
order. Both bulk paths now capture the flag before the mirror refresh.

**Still open, deliberately:** the tier hands both spans one unified `count`, so a block that produces more than
it consumes is bounded by the input count; separate in/out counts are the next step. Tag slots are fixed at 64 x
512 B per dispatch, and overflow is reported rather than silently dropped.

### 76.1 Stage 1 — consume/publish and tag READING, no new mechanism

A device span pair satisfying the full concepts, whose `consume`/`publish` record into device memory. After the
kernel the host reads those counters and calls `consume`/`publish` on the real port spans — at which point
**`blockManagedIO` already does the rest** (`Block.hpp:1969`, "honour an expert hatch's own consume()/publish()
exactly as the CPU processBulk path does"). The seam for this already exists; only the counter transfer is new.

Caveat to verify before relying on it: reading a tag's _index_ in a kernel is certain (POD). Reading its
**payload** through `ValueMapView::get_if` is plausible-but-unproven on device — `DeviceLog` proves the WRITE path
(`try_emplace`) in a kernel, not the read path. Probe it before promising it.

### 76.2 Stage 2 — stateful bodies, and the cost is not what the old prohibition assumed

Every device tier requires a `const` body because a kernel body is shared by every work item. **But the framework
bulk tier launches ONE work item** (`parallelFor(ctx, 1UZ, …)`, `ExecutionStrategy.hpp:251`), so a non-const body
there cannot race — its writes are merely discarded, because `relocateBlockToDevice` is one-way by design.

Copying the mirror back would fix that, and the cost is **`sizeof(Block)` once per dispatch, not per sample**.
For a ~3 KB block against a 65 536-sample dispatch that is noise — a completely different cost class from the
per-sample read-back §66 measured at ~17 ns/sample. **So "no mutable state on a device" is a correct rule for the
per-element tiers and an unnecessarily strong one for the single-work-item bulk tier.**

This is what would let a stateful block — a zero-crossing counter, a running integrator — be a real kernel body.

### 76.3 Stage 3 — tag PUBLISHING from a kernel

The only stage needing genuinely new machinery, and §74's reserve/fill/publish is it: the host reserves N tag
slots and their blob storage before the launch, the kernel fills pre-assigned slots via `ValueMapView::try_emplace`
(already proven in `DeviceLog`), the host publishes the actual count afterwards. The concept's
`publishTag(property_map&, …)` cannot be honoured verbatim on device — an overload taking a **non-owning
`ValueMapView`** is what a kernel can call.

### 76.4 Why this ordering

Stages 1 and 2 need no new mechanism and no new buffer type; they are adapters over machinery that already exists,
and together they let a classical `processBulk` body run on a device for everything except publishing tags. Stage 3
is the only part that adds a mechanism, and it is exactly the part the maintainer's SIMO observation already
simplified: the host does the single-producer claim, the kernel fills pre-assigned slots, so ordering falls out
without a sort.

## 75. SETTINGS ON A DEVICE — read yes, write no, and two defects found while checking (2026-08-26)

### 75.1 Reading works, and is now measured rather than asserted

`scratchpad/settings_probe.cpp` on `gpu:sycl`:

```
Q1 after init+run : taps.data()=0x7f78... residency=shared size=2 epoch=1
```

A `std::pmr::vector<float>` setting really is in **shared USM** after `init()`, so the kernel indexes the same
buffer the host owns. The mirror refreshes when `_settingsEpoch` moves (`ExecutionStrategy.hpp:192`), so a
host-side change is picked up on the next dispatch. The documented constraints hold: fundamental or trivially
copyable, pmr containers of trivially copyable elements, or ports — no strings, because SSO keeps short data inside
the object and a device copy would point back at the host.

### 75.2 Writing settings FROM a kernel should not be supported — three independent reasons

1. **Ill-defined under data parallelism.** N work items, one field. Without a reduction discipline the result is
   whichever work item happened to land last. GR4 settings are single-writer by construction.
2. **The relocation is one-way on purpose.** `relocateBlockToDevice` copies host→device and nothing comes back
   (`ExecutionStrategy.hpp:193`). Making it two-way costs a device→host copy of the block per dispatch — precisely
   the ~17 ns/sample read-back §66 spent the effort avoiding.
3. **It breaks settings coherence.** Settings are reflected, SigMF-serialisable, and drive tag and message emission
   on the host. A device writer has no defined ordering against a concurrent host `set()`, so the block's
   externally visible state would stop being a function of the host's own timeline.

**What people actually want is not settings modification — it is device→host reporting** (an adapted coefficient,
a running statistic). That is an _output_. Use the same slab pattern as §74: the kernel writes POD records into a
USM slab, the host drains after the barrier and applies them through the ordinary settings path, so the epoch,
reflection and message emission all stay intact. **One mechanism serves both tags and settings feedback — build it
once.**

### 75.0 BOTH DEFECTS FIXED (2026-08-26)

**The canary was worse than "once per type".** In the non-shadow branch the mirror is rebuilt on EVERY call
(`ctx.allocateShared<TBlock>(1)`, `ExecutionStrategy.hpp:199`), so the canary's own contract — _runs when the
mirror is about to be (re)built_ — demands it fire every time. The `static std::atomic_flag` fired it once per
type, ever. Now `return true` for that branch, matching the invariant; `<atomic>` is left unused in the file.

**The over-claiming assertion is honest, and a real test replaces the claim.** `qa_DeviceAutoParallel` gained _"a
pmr setting changed mid-run keeps its device seat and the kernel reads the new values"_: a `std::pmr::vector`
setting changed **by tag while the graph runs** — the documented path — asserting the allocator is still the
device resource afterwards AND that samples before/after the change use the old/new taps. **It passes** (42
asserts / 5 tests, was 30 / 4), so the docs' promise is true and now pinned. The old assertion's message was
corrected to what it actually checks ("the seat survives the run").

### 75.3 Defect: the mutation canary fires once per TYPE, not per block

`isFirstUseOfTheseSettings` (`ExecutionStrategy.hpp:207`) keys on the settings epoch when the block owns a device
shadow — correct. In the **non-shadow** branch (functors outside the `Block<T>` hierarchy) it uses

```cpp
static std::atomic_flag probed = ATOMIC_FLAG_INIT; // function-local
```

so the first instance of a given type is probed and **every later instance never is**. The comment claims
"mutability is a property of the type", which is true of the _type_ but not of the guard's purpose — the canary
also catches a block whose settings changed. Narrow blast radius (that branch is trait-test functors, not real
blocks), but it is a guard that silently stops guarding.

### 75.4 Defect: a test asserts something it does not exercise

`qa_DeviceAutoParallel` has

```cpp
expect(dut.taps.get_allocator().resource() == deviceMr) << "a later settings assignment keeps the device seat";
```

**There is no later settings assignment in that test** — `taps` is assigned once, before the run. The assertion
checks the allocator after a single seating and its message overstates what it proves. The user-facing docs promise
("Settings assignment keeps the device seat … You do not have to think about it") rests on it.

My probe could not confirm the promise either: a `set()` + `applyStagedParameters()` after the scheduler stopped
staged cleanly (`rejected 0`) but applied nothing (`applied 0`), so the interesting path — a settings change while
the graph runs — is **still unverified**. This is the §40.5 category again: a test that would pass for the wrong
reason. **Pin it with a message-driven settings change mid-run, or weaken the docs.**

## 73. ROUTE A DECIDED — hoist the members' shared domain onto the group (2026-08-26)

**Maintainer decision:** whether it names a thread pool or a device, the memory is tightly linked to the common
`compute_domain` of a sub-graph's members, so the group block should carry it. Route A over Route B (propagating
`PortMetaInfo` through `exportPort`).

**Measured before agreeing, and two of my own claims died.**

### 73.1 The thread-pool objection does NOT exist — retracted

`scratchpad/routeA_probe.cpp`, Q1: a flat graph whose middle block carries the given `compute_domain`.

| compute_domain | terminated | wall   |
| -------------- | ---------- | ------ |
| _(default)_    | yes        | 5.4 ms |
| `host`         | yes        | 0.6 ms |
| `no_such_pool` | yes        | 0.5 ms |
| `gpu:sycl`     | yes        | 91 ms  |

**A `compute_domain` naming no registered pool does not stall a block.** The scheduler picks its pool from its own
`poolName` setting (`Scheduler.hpp:262`) and, on an unknown name, keeps the existing pool and emits an error —
it never hangs. The comment in `DeviceSubGraph.hpp` claiming _"asks for a pool nothing registers, and the block is
then never scheduled — a graph that spins forever"_ is **FALSE and is in the branch**; the hang it was written for
had some other cause, never found. **Fix the comment.**

### 73.2 The real constraint: the hoist cannot come from outside the wrapper

Q2, `source -> groupA -> groupB -> sink`, both groups homogeneous on `gpu:sycl`, domain staged on each wrapper via
`BlockModel::settings().set()`:

```
group domain=gpu:sycl  ran=yes samples=2048  [src->A host] [A->B host] [B->sink host]
[diag] staged compute_domain on the wrapper: ABSENT — settings().set() through a BlockModel did not stage it
```

**§44's trap again** — the mechanism that made `disconnect_on_done` a no-op for its entire life. So Route A must
set the domain **inside `makeSubGraph`, on the concrete `SubGraph` before type erasure**, never from outside.

Second constraint: `applyEdgeConnection` reads `settings().stagedParameters()` (`Graph.hpp:722`), so the domain
must still be **staged** when the parent edges are connected — applying it first erases it.

### 73.3 Where the maintainer's premise needs a guard

`refuseTwoDeviceDomains` permits **host members alongside one device domain**, so "all members share a domain" is
not enforced. Two cases where hoisting would be wrong:

1. a group with a **host member on its boundary** would claim device memory for a port fed by a host block;
2. a group built by `makeDeviceSubGraph` **deliberately owns its host boundary** — its boundary members are the
   transfer blocks. Hoisting there marks the parent edge device while the transfer expects host input, silently
   moving the crossing outward and making the transfer a no-op.

**Rule: hoist only when every member declares the same device domain AND no boundary member is a transfer block.**
Host→group edges correctly stay host either way (measured) — Route A only changes group→group.

### 73.4 Route B is not dead, it is the mixed-type answer

`PortMetaInfo::data_type` already carries `gr::meta::type_name<T>()` (`Port.hpp:588`) and `Port.hpp:1266` already
uses it to refuse a type mismatch on connect. If an exported port inherited the inner port's `metaInfo`,
`makeDeviceSubGraph`'s `T` parameter would disappear and mixed-type boundaries would work. **Start by root-causing
the segfault §64 records** from the earlier attempt to compare `metaInfo.data_type` — do not just retry it.

## 74. TAGS ON A SYCL DEVICE — re-evaluated, and the answer is better than the record said (2026-08-26)

The old note called this "blocked". That is too strong. Split it three ways.

**Generation from a kernel: FEASIBLE TODAY, and the pattern already ships.** `DeviceLogSlab` (`DeviceLog.hpp:66`)
does exactly the required shape: a fixed-size slot array in USM, a `gr::atomic_ref(...).fetch_add` claim-or-drop, a
kernel writing packed records, the host draining after a barrier and merging. A device tag path is the same
machine: **the kernel emits `(sampleIndex, payload)` POD records into a device tag slab; the host drains after the
kernel completes, sorts by sample index, and publishes through the ordinary `publishTag`.** This dodges all three
historical blockers at once — no `atomic_flag` on device, ordering restored by the host sort, and no ValueMap blob
is ever read on the device.

**Publishing directly into the tag ring from a kernel: DO NOT.** `publishTag` (`Port.hpp:1076`) calls
`storeBlob` → `ChunkBuffer::serialiseBlob` (`ChunkBuffer.hpp:139`), which opens with

```cpp
while (s._guard.test_and_set(std::memory_order_acquire)) { }
```

a spin lock. Thousands of work items contending on one flag is a forward-progress hazard on a GPU, not merely
slow. And a `fetch_add` claim cannot produce index-ordered insertion, which tags require.

**Reading tags in a kernel: partially feasible.** The tag axis is already forced host-accessible
(`Access::Shared`, `Graph.hpp:765`) and `Tag` is trivially copyable by static_assert (`Tag.hpp:93`), so a **POD
projection** — occurrence, sample index, a few numeric fields — is reachable now. The full `property_map` is not:
variable-length, and the `kBlobAlignment` issue below.

**Capability check:** `processBulk_sycl` already has full tag access because it runs on the host thread. What is
missing is only the framework-managed auto-parallel tier.

### 74.1 `kBlobAlignment`, stated correctly (my earlier note was wrong on the number)

`ValueMap` guarantees a 16-aligned blob base (`kBlobAlignment = 16`; `_alignedAllocate` over-allocates and stores a
recovery byte to force it; `from_blob` asserts it), and records inside a blob are 16-aligned too
(`kRecAlignment = 16`, `alignToRecord`) — **but only relative to the blob start.**
`ChunkBuffer::serialiseBlob` packs blobs back to back: `s.headOffset += need`, with no rounding. So the second and
later blobs in a chunk start wherever the previous one ended. Misalign the base and every record inside is
misaligned; the read path casts without validating. x86 tolerates it, a GPU does not.

The old note said "packs at 8-byte at best". **Wrong: there is no alignment guarantee at all.** Substance held,
number did not. Gate when the by-value tag route is attempted: assert `tag.map._blob % 16 == 0` on a device edge —
it will not hold today. Belongs to the tag branch; latent here because the tag axis is host-accessible.

## 72. NOMEN-EST-OMEN PASS — comments, names, STL (2026-08-25)

Two passes over everything this branch adds, under the §0.6/§0.7 rules.

**Pass 1, comments: 1022 -> 731 added comment lines.** Paragraph-length rationale moved to the commit messages
where it already lived; class-level `@brief` trimmed to what the type is. Distributed across the 16 commits by
whichever commit introduced each line.

**Pass 2, necessity + naming + STL (host code only; kernel-callable code is exempt because most `std::ranges`
algorithms are not device-callable under AdaptiveCpp SSCP).** 731 -> 717, but the line count understates it: the
point was deleting comments by making the code say the thing.

| finding                                                                                                                                    | disposition                                                                                                                                                           |
| ------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `SubGraph::_hasCycle` was **write-only dead state** — set, reset, read nowhere; its comment described the chain machinery deleted under D5 | field and both writes removed                                                                                                                                         |
| `makeSubGraph`'s `@brief` had drifted onto `struct BoundaryPort`, 60 lines from what it documents (caused by pass 1's own line-range edit) | moved back                                                                                                                                                            |
| `qa_FFTSubGraph` still described `bindChainBuffers()` registering a `DomainBinding`                                                        | both gone since D5; comment rewritten                                                                                                                                 |
| `topologicalOrder`                                                                                                                         | `inDegree` -> `unresolvedPredecessors`, `emitted` -> `ordered`, inner rescan -> named `releaseSuccessorsOf` lambda, trailing loop+comment -> one `ranges::copy_if`    |
| `refuseTwoDeviceDomains`                                                                                                                   | nested loop -> `views::transform \| views::filter` into a `std::set`; a local `namesThreadPool` replaced the comment explaining why "host" and the pool ids are alike |
| `portNameAt`                                                                                                                               | one caller: folded in (rule 7)                                                                                                                                        |
| `makeDeviceSubGraph`                                                                                                                       | two near-identical 14-line loops -> two named lambdas, `uploadInFrontOf` / `downloadBehind`                                                                           |
| `DeviceContextRegistry::lookupLocked`                                                                                                      | -> `longestRegisteredPrefixOf`; the comment describing prefix-shortening is now the name                                                                              |
| 11 comments whose symbol already said it                                                                                                   | deleted (`invokeProcessBulkSycl`, `blockManagedIO`, `allUserMembersTriviallyCopyable`, `firstNonRelocatableMember`, `parallelFor`, `if constexpr` branch markers)     |

**Judged against the rule, deliberately:** `appendArg`'s `if/else if` chain over `get_if<T>` stays. The STL form
is a fold expression wrapping an immediately-invoked lambda, needing `.template operator()<...>` at the call
site — worse than five lines anyone reads at a glance. "Use the STL" is about clarity, not about reaching for
`std::visit`.

**Kept on purpose:** class-level `@brief` where the type is not self-evident, the G10 constraint (AdaptiveCpp
aborts if a kernel launches from `~runner`), the `SubGraphVertical` topology diagram, the `@code` protocol blocks
whose _ordering_ is the contract, and ~17 comments that exist to say **why the obvious STL call is wrong**
(`llrint` is an unresolved extern under SSCP; a denylist rather than a `data()`/`size()` shape test, which would
also catch `std::array`). Delete those and the next reader "simplifies" the device build into breaking.

### 72.0 `qa_Graph` FALSE ALARM — the registry-off build dir, not a regression

`qa_Graph` came back rc=255 with 8 failures in `Graph::groupBlocks` / `Graph::ungroupBlocks`, every one
`cannot spawn sub-graph/scheduler of type 'gr::scheduler::Simple<>'`. That is `core/src/Graph.cpp:138`, reached
when `pluginLoader().instantiateScheduler()` returns null — a **block-registry** lookup. `build-acpp` has
`GR_ENABLE_BLOCK_REGISTRY=OFF`, and `core/src/Graph.cpp` is touched by zero commits on this branch.

**Proved rather than argued:** the same test in `build-acpp-registry` (registry ON) is rc=0 —
`Graph::ungroupBlocks` 70/5 and `Graph::groupBlocks` 129/8 both green.

**The trap, restated because it keeps costing time:** `build-acpp` is registry-OFF, so any suite whose path goes
through the plugin loader fails there for environmental reasons. Use `build-acpp-registry` for those. This is the
same shape as the older note that `blocks/fourier/test` is never configured under a registry-off acpp dir.

### 72.1 THE MECHANISM WAS THE BUG — ten defects, none of them judgement

Bulk `(file, startLine, endLine, replacement)` tables produced, in order: a doubled `/*`; a dropped `*/`; a range
that swallowed the `};` closing a struct; a line inside `stopDispatch()` replaced by that function's own
signature; a grep anchor that matched prose an earlier edit had also written into a `@brief` 50 lines away; a
duplicated `println`; a duplicated `auto backends`; a `@brief` inserted **six times** because the replacement
contained its own search text; blank-line residue from deletions that did not swallow their newline; and a
deleted `_quiescent` declaration from two rows interacting. Every guard added exposed the next.

**Two of them surfaced as errors deep inside CUDA and libstdc++ headers** (`using declaration in class refers
into 'std', which is not a class`) — which points at the toolchain, not the edit. An unclosed brace makes the
compiler parse `cuda.h` inside a class body. Confirmed mine by rebuilding the same target against the original
header, which passed.

**What actually works, and is what this branch was distributed with:** treat the verified working tree as the
source of truth, `git diff` it, split that diff **per hunk**, and let `git apply --check` place each hunk at the
first commit whose content it fits — which is the commit that introduced that code. Idempotent by construction
(an applied hunk no longer applies), exactly attributed, no conflict residue. 23/23 hunks placed; the resulting
history delta is byte-identical to the verified tree.

Three tooling rules that came out of it, now §0.8:

- anchor a range on the **declaration line and walk brace depth**, never on comment prose;
- the delimiter guard compares **deltas against `HEAD`** — absolute `{}`/`()` counts are unreliable because
  literals contain delimiters, and the first version gave two false positives I nearly acted on;
- **strip ANSI before grepping a build log**: `grep -c ' error:'` reported zero while the log held two real
  errors, because the colour escape sits between the space and `error`. Twice this session I reported green
  suites off stale binaries because of it — test runs are now gated on a clean log.

## 70. THE DE-SHADERING SWEEP, COMPLETED (2026-08-25)

D5 removed the GLSL and WebGPU _code_; a grep showed it had left the _vocabulary_ behind in sixteen places across
thirteen files, including two that were user-visible or genuinely dead:

| remnant                                                                                                                              | why it mattered                                                                                                                 |
| ------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------- |
| `Block.hpp` runtime warning naming a `shaderFragment` hatch                                                                          | told a user, at run time, to write a method the framework no longer looks for anywhere                                          |
| `BackendCompat.hpp`: `GlslProgram` + `kInvalidGlslProgram`                                                                           | dead types, zero references                                                                                                     |
| `SubGraph.hpp`: a 9-line comment on chain partitioning                                                                               | documented code deleted in §58.4; it hung above an unrelated function, describing nothing                                       |
| `device_test_helpers.hpp`: `// GL compute shader test: …`                                                                            | a declaration comment with no declaration under it                                                                              |
| `bm_FFT_backends_helpers.cpp`: `// ── WebGPU backend (browser only) ──`                                                              | an empty section header between two real sections                                                                               |
| comments and doc strings in `DeviceContext.hpp`, `ExecutionStrategy.hpp`, `fft.hpp`, `DeviceExpectation.hpp`, `qa_DispatchGroup.cpp` | described the branch's backends as including shader ones                                                                        |
| `qa_SubGraph.cpp`: `gpu:glsl` as the second device domain                                                                            | now `gpu:cuda` — still an unserved domain, which is all the test needs, and it names a backend the branch actually targets next |

**Fixed by amending the commits that introduced each line**, not by a follow-up commit: 4 (runtime contract), 5
(dispatch seam), 6 (sub-graph), 9 (FFT) and 12 (transfer insertion, which had moved one of the comments). The
history therefore never contains the shader vocabulary at all. Delta over the whole branch: 13 files,
+30 / −50, comments and dead typedefs only — **no behavioural change**.

`git grep -iE "shaderfragment|webgpu|wgsl|glsl|glcompute|spirv|shader|opengl|\bEGL\b"` over the tracked tree now
returns **nothing**. The three surviving `Asyncify` hits are `blocks/sdr` and the top-level `CMakeLists.txt`, all
present on `origin/main` and unrelated to this work.

**Second miss, caught on review:** the replacement for the `OpenGL 4.3 …` example on `DeviceContext::version()`
was _invented_ (`"AdaptiveCpp 25.02 / CUDA 12.6"`). `version()` returns `sycl::info::device::driver_version`,
which is a bare `"13030"` on this card and `"1.2"` for the OMP host device. A fabricated example comment is the
same class of defect as the `shaderFragment` warning being fixed — **run the getter before writing what it
returns.** Now `// the backend's own driver-version string, e.g. "13030"`.

**Trap:** a rebase that touches a file a later commit _refactors_ will conflict, and `git checkout --theirs`
silently discards the edits just made — resolving the conflict took the incoming version of `SubGraph.hpp`
wholesale and reverted four fixes. Re-apply after resolving, then grep the file again before staging it.

## 71. TWO ITEMS DROPPED FROM SCOPE (maintainer, 2026-08-25)

**Sub-graph auto-formation.** A helper: it needs no core adjustment and can be a later addition or its own PR.
Tracking it was pure carrying cost, so §38's deferred design is gone and it is off the ToDo list. Explicit
membership through `makeSubGraph` / `makeDeviceSubGraph` is the supported way to form a group.

**`processEpilogue` on the device path.** It exists on the CPU for two reasons that are both CPU-shaped: SIMD
batch sizes, and tags splitting processing into minimum-sample frames. Neither applies to a device, where work is
submitted in bulk and the framework already handles the boundary. Adding it would cost without adding value; if a
real need appears it can be added then. Removed from the records rather than carried as a permanent open item.

## §77 · CUDA VMM removed, and what a kernel can actually do with a tag

**§77.1 device-only memory: SYCL has it, the ring needs more.** `sycl::malloc_device` is device-only memory, so
the earlier "CudaVmmMemoryResource is the only provider" line was wrong — what VMM uniquely gave is
double-mapping. `CircularBuffer` fatals on `deviceOnly && !usesMMAP`: a wrap needs either the host mirror copy
(which faults on device-only memory) or the same pages mapped twice. No portable SYCL API maps physical pages
twice; AdaptiveCpp does not implement `sycl_ext_oneapi_virtual_mem`. Removed the provider (~450 lines, 13 files) —
with the opt-in off, which is the default and every CI lane, it already returned `nullptr` and fell through to
shared USM, so nothing anyone builds changes. Vocabulary and capability predicates kept; tests that asserted on the
_resource_ retargeted to the _domain_, which is what they meant.

**§77.2 the tag read path splits.** Measured on the GPU, link-level and run-level: `size`, `contains`, `find`,
`keys()`/`keyId()` all run in a kernel — canonical keys match by integer id, so no string compare. `get_if<T>` /
`at` / iterator deref do NOT: they route through `ValueView::get_if<T>` and `monostateRecord()`, explicitly
instantiated in `Value.cpp` and therefore host-only symbols. **A kernel can detect and classify a tag; it cannot
read its values.**

**§77.3 input tags still do not reach a kernel — and the first attempt was structurally wrong.** `rawTags()` is a
lazy `views::transform`, so there is no contiguous `Tag*` to hand a kernel, and the ring it projects is not
device-accessible when the producer is a host block — which is exactly the case worth serving. Reaching a kernel
needs descriptors AND blobs staged into aligned device slots (the output side already has that shape). Left
unwired with an honest comment rather than half-built. Open decision before building it: given §77.2 bounds the
payoff to presence/classification, is staging worth the per-dispatch blob copy, or is a cheaper
`{sampleIndex, keyId-set}` projection the better trade?

**§77.4 process note.** Two green-looking results were wrong this round: a test run off a stale binary after a
failed build (the `check.sh` gate exists for this — do not invoke a test binary directly), and a probe that
printed all-zeroes because the host-side map build had silently failed. Both were caught only by checking the
build log and the setup preconditions first.

## §78 · Current state and what is actually open (2026-08-27) — SUPERSEDES every earlier ToDo list here

**18 commits.** Order: four `fix(core)`/`fix(sdr,timing)` commits carrying pre-existing defects that are
cherry-pickable ahead of the feature work, then the device arc, then three late refinements that genuinely belong
where they sit. Three commits that only undid earlier work on this branch were folded away; two more were kept
deliberately (a measured perf change, and a fix whose diff depends on a later restructuring).

**Green:** full build 0 errors; every device suite, `qa_Tags`, `qa_Block`, `qa_buffer`, `qa_ValueMap`,
`qa_SubGraph` pass; GCC 15 and Clang 20 clean.

**The three red `ctest` entries, each characterised — do not chase them as branch regressions:**

- `qa_Graph` — environmental: `build-acpp` has `GR_ENABLE_BLOCK_REGISTRY=OFF`; passes with it ON.
- `qa_DataSetEstimators` — pre-existing: identical failure at the merge-base (`28 ~ (33 +/- 1)`).
- `qa_SubGraphAssets` — pre-existing, and it _hangs_ on a blocking `schedulerThread.get()`. It does not even
  compile at the merge-base under GCC, so `main` is red for that target too.

**Closed since the last entry:** device spans completed to the real span surface (they were an incomplete backing
implementation and broke the acpp CI lane); input tags now reach a kernel, staged into aligned slots; nested maps
buildable in a kernel; `WireMapLike` removed; three header merges; 35 discarded-`std::expected` sites fixed.

**SUPERSEDED at the top: R0 (device-private block state) now precedes everything below.**

**Actually open, in the order the maintainer chose:**

1. ~~harden `qa_DeviceAutoParallel`~~ — done. All four dispatch points assert zero fallbacks, mutation-tested.
2. ~~refresh this file~~ — done (this section).
3. **Kernel-side value extraction — DESIGN ONLY for now.** `ValueView::get_if<T>` is explicitly instantiated in
   `core/src/Value.cpp`, so a kernel can detect and classify a tag but cannot read its values.
4. `qa_SubGraphAssets` hang — pre-existing, orthogonal, worth its own investigation.

**Precise semantics of `gr::test::cpuFallbacksDuring`, learned by mutation:** zero means _the dispatcher did not
refuse a kernel_, NOT _the block was dispatched_. A block on a domain naming no device never reaches dispatch and
scores zero trivially; `gpu:sycl:99` resolves to the canonical device via index fallback and genuinely runs. The
mutation that actually exercises it is a device kind with no backend registered, e.g. `gpu:cuda`.

## Action plan — A/B/C (drawn up 2026-09-09, branch at 24 commits)

Everything below is evidence-backed; where a claim is unverified it says so. Nothing here is started.

### A · Reduce complexity, clean up the new API

**A1 · A hatch escapes the domain policy it should obey.** `FastConvolution.hpp:86-90` and `Reduce.hpp:65-69`
return the host result when a device allocation fails — silently, no warning. A block spelled `gpu:sycl!` still
computes on the host. The framework guard cannot see inside a hatch, so the hatch must report the refusal
itself. **~30 SLOC, and it is a correctness hole in a policy that just shipped.**

**A2 · `processBulk_sycl` names a backend in a backend-neutral framework.** Renaming it (`processBulkDevice`)
and passing `DeviceContext&` rather than `SyclQueue&` is cheap now and impossible once anyone outside this
branch writes one. `algorithm/Reduce.hpp:55-64` takes a raw `sycl::queue&` for the same reason. **~80 SLOC.**

**A3 · Four tiers are selected by qualifier spelling.** `const noexcept processOne` vs `const processBulk(view)`
vs `const processBulk(span)` vs the hatch — an author discovers which tier they landed in by its behaviour. A
declared opt-in (a trait or alias) would say it. **Design first, then ~120 SLOC.**

**A4 · Three ways across the boundary, one documented.** `HostToDevice`/`DeviceToHost` blocks, `makeDeviceSubGraph`,
and residency-by-edge-domain. The first two are registered/documented user API so they are not dead code, but
having three is a choice nobody has made deliberately. **Decide, then delete or document.**

**A5 · Public `_data`/`_size`/`_acct` on `DeviceInputSpan`** (`DeviceSpans.hpp:42-48`) — underscore-private names
in the public API a kernel body touches.

### B · Performance against the theoretical limit

Measured on this machine, RTX 3070 + 5900X. **Unverified by a profiler — all splits are arithmetic from wall
clock and pass counts. Treat as direction, not as numbers to quote.**

**B1 · Per-dispatch USM allocation, ~92 µs.** Nine shared regions allocated and freed every dispatch
(`ExecutionStrategy.hpp:262,401,424,482,501`). A pool would take this to ~0. **Largest single win, and it is
framework-side so every block gets it.**

**B2 · Staging goes through page-migrating managed memory**, not device memory: pinned ring → memcpy → managed
scratch → kernel → managed → memcpy → pinned. `SyclFFT` already stages into `allocateDevice` and gets the
documented 15x; the framework tiers do not. **This is the branch's own earlier finding, not applied to the
tiers.**

**B3 · At least three full queue syncs per dispatch** (`DeviceContextSycl.hpp:123-125,146,159` plus
`pollDeviceError`), so nothing is ever in flight across `work()` calls.

**B4 · The FFT block is ~90 % PCIe-bound** (335 µs transfer against 75 µs kernel at N=4096 x128) — 0.4 % of fp32
peak. The backend benchmark reaches 1.4 GS/s where the block reaches 13-30 MS/s; the gap is B1-B3, not the
transform.

**B5 · The window tier on a GPU is one work item per window** (`ExecutionStrategy.hpp:802-804`): 1 Mi samples at
a 64 ki frame is 16 threads. Offer that tier on a device only when the window count is large.

**B6 · Several reported wins are noise.** `bm_DeviceDispatch` 64 ki rows swing -47 %..+39 % run to run because
they time graph construction and sink logging. Only the >=1 Mi rows are stable. Fix the harness before any
number is quoted.

### C · Spikes against the guidelines and the block contract

**The contract to hold every block to: host-native, `host:sycl` and `gpu:sycl`, each numerically correct, with a
sensible default and configurable.** Current state against it:

| block                            | host | host:sycl | gpu:sycl | gap                                                                  |
| -------------------------------- | ---- | --------- | -------- | -------------------------------------------------------------------- |
| `FFT<float>`                     | yes  | yes       | yes      | —                                                                    |
| `FFT<double>`                    | yes  | **no**    | **no**   | `SyclFFT` is float-only (R2)                                         |
| `FastConvolutionFilter`          | yes  | yes       | yes      | silent host fallback (A1); re-uploads taps every call                |
| `RationalResampler`              | yes  | yes       | yes      | —                                                                    |
| `DriftResampler`                 | yes  | **no**    | **no**   | per-sample recursive body, host by design — state it                 |
| `Correlator` / `FrameStatistics` | yes  | yes       | yes      | —                                                                    |
| `Channeliser` (test block)       | yes  | yes       | yes      | 1:1 only; a decimating polyphase bank needs a window on a collection |

**C1 · No tag-propagation test exists for any new block.** `Correlator`, `DriftResampler`, `FastConvolutionFilter`,
`RootMeanSquare`, `RationalResampler`, `FFT`. CLAUDE.md §8 requires it.

**C2 · ~14 device tests return with no assertions on a GPU-less runner.** Now that `!` exists they should demand
the domain and fail. This is the item that makes every other claim checkable, and it is cheap.

**C3 · Two blocks reflect a `std::vector` setting** (`FastConvolution.hpp:35` taps) against the doc's own advice.

**C4 · `settingsChanged`-derived non-reflected state is frozen on the device** after the first dispatch
(`DeviceRelocatable.hpp:148-162` copies reflected members only). Silent wrong numbers; no test covers it.

### Order

**C2 first** — until the device tests fail rather than skip, nothing else is verifiable. Then **A1** (a shipped
policy with a hole), then **B1/B2** (the framework-side performance wins that every block inherits), then **A2**
(cheap now, expensive later), then C1/C4, then the rest.
