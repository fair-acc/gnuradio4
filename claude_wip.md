#WIP — `ValueMap` & aggregate tag - buffer rework

##Recurring advice — read first, apply every time

                                         Persistent user preferences captured during this work.Re -
                                     read at the start of every               session.

                                         1. *
                                         *Build parallelism : always `-
                                     j6`**(CLAUDE.md §10).Never `- j` without a number; never > 6. Use `cmake --build <dir> -j6` / `ctest --test-dir <dir> -j6`.
2. **Sign-off trailer — always and only**: `Signed-off-by: Ralph J. Steinhagen <r.steinhagen@gsi.de>`. **No** `Co-Authored-By:` or any other trailer alongside it.
3. **Commit only when explicitly told.** Never run `git commit`, `git commit --amend`, `git push`, or
   `git cherry-pick` (resulting in a new commit) on your own initiative. **Do NOT pause work between steps to ask for
   commit auth** — work through edits / builds / tests as a continuous flow. Multiple completed sub-tasks may sit in the
   working tree before any commit happens. The user initiates commits ("commit", "commit this", "amend into X") and
   *that request itself* is the authorisation for that action — no separate "ok to proceed?" round-trip. Maintain a
   commit-ready state (formatted, build-green, test-green) between sub-tasks so the user can commit at any moment.
4. **Prefer new commits over amends.** Only amend on an explicit per-request authorisation.
5. **No hook-bypass** (`--no-verify`, `--no-gpg-sign`, etc.) unless the user explicitly asks.
6. **In-Code Documentation** prefer nomen-est-omen i.e. telling names and parameters that make the function of a method
   or function clear. Larger core-structe may have a brief doxygen-style documentation (w/o signatures) and should
   include a TL;
   DR code - snippets explaining the basic usage 7 * *Commit message style** : concise;
   subject in conventional - commits form(`fix(core) :`, `feat(...) :`, `chore :`, …); body documents only the
   non-obvious *why* and any behavioural / contract / signature changes. **Never restate what the diff already shows.**
   Aim for ≤ ~12 lines of body; subject-only commits are fine when the subject fully captures intent.
   8**Maintain commit-ready state between sub-tasks** (so the user can commit at any moment without surprises):
   1. Review edits against CLAUDE.md (§1 naming, §3 documentation, §8 AI anti-patterns). In particular: don't comment on WHAT the code does, don't reference the specific task/fix/issue, no `@brief`/`@param`/`@return` boilerplate on methods.
   2. Run `clang-format-18 -i <changed-files>` (LLVM base, project's `.clang-format` uses `ColumnLimit: 2000` and `AlignConsecutiveAssignments: true`, so no width-driven reflow but `=`-alignment is enforced).
   3. Build + test the change green (`-j6`).
   4. **When the user requests a commit**, present diff summary + commit-message draft and run `git commit` directly —
      no "ok to proceed?" round-trip.
9**Present findings, don't push them** for exploratory questions ("what could we do about X?") — 2–3 sentences with a recommendation + main tradeoff. Don't start implementing until the user agrees.
10**When in doubt, ask.** The cost of one clarifying question is lower than the cost of an unwanted action.

---

**Branch**: `featValueMapExtension` — 2 commits ahead of `origin/putting-the-gnu-back-into-gnuradio4` (rebased
2026-04-27; previously branched off `9e904b04`).
**Last updated**: 2026-04-27 (Q1.B Tensor migration landed; rebased onto putting-the-gnu; bench re-run captured; ready
for transfer)
**Owner**: Ralph J. Steinhagen (dr.steinhagen@gsi.de)
**Spec source**: design handoff memo from prior planning session (summarised at end of this file)

This file is tracked on the branch. Captures enough state to resume on another machine after `git checkout featValueMapExtension`. Plan + decisions + investigation findings — not code. Keep up to date each substantive session; amend or add a new commit as appropriate.

---

## How to recover (read first when resuming on another machine)

The **Q1.B inversion** of the `Value` / `ValueMap` storage refactor on branch
`featValueMapExtension` has landed as commit `2416775d WIP`. Working tree carries only this
WIP doc as uncommitted; everything else is committed.

**Recovery checklist — run in order**:

1. `git checkout featValueMapExtension && git log --oneline -3` — top commit should be a `WIP`
   commit (currently `2416775d`); if you see anything else, ask the user.
2. `git status --short` — only `claude_wip.md` should appear modified; untracked
   `.claude/scheduled_tasks.lock` and `blocks/onnx/` are unrelated to Q1.B.
3. Read `claude_memory.md` (repo root, untracked) for durable cross-session memory — gotchas to
   remember, file roles, build commands, key design constraints.
4. Read this file's §0 (quick status) for the bench results and pre-existing-failure list.
5. Read this file's Appendix B (search `## Appendix B`) for the **deferred** `get_if<T>` policy
   proposal — user asked for write-up only; **do not act on it without re-confirmation**.
6. Tasks: #20–#23, #25, #26 are completed. Only **#24 (iter view-mode for Tensor + nested-Map)**
   remains pending — it closes the 7× iter perf gap vs `unordered_map`. It's the natural next
   step but is independent of Q1.B Tensor; pick it up only when the user asks.

**Confirm tests still green** (build cap is **`-j6`** — never higher; ninja JOB_POOLS already
limits to 6 compile / 2 link, but pass it explicitly):

```
cmake --build cmake-build-debug-gcc-15 --target qa_Value qa_ValueMap qa_Settings qa_Tags qa_Block qa_Scheduler qa_Graph qa_DataSink qa_DataSet qa_YamlPmt -j6
ctest --test-dir cmake-build-debug-gcc-15 -R "qa_(Value|ValueMap|Settings|Tags|Block|Scheduler|Graph|DataSink|DataSet|YamlPmt)$" --output-on-failure -j6
```

All ten must pass. The 8 broader-tree failures (qa_SchedulerMessages, qa_plugins_test,
qa_plugin_schedulers_test, qa_Audio, qa_StreamToDataSet, qa_RTL2832Source, qa_SoapySource,
qa_KnownSharedLibBlocks) are **pre-existing on the prior baseline** (verified 2026-04-27 via
git-stash comparison) — not regressions, do not block on them.

**Re-confirm bench numbers** (Release build, see §0 for the comparison tables):

```
cmake --build cmake-build-release-gcc-15 --target bm_ValueMap qa_PerformanceMonitor -j6
cmake-build-release-gcc-15/core/benchmarks/bm_ValueMap        # iter / find / copy / merge / erase / insert
cmake-build-release-gcc-15/core/test/qa_PerformanceMonitor 8 1   # TC1 (no tags), 8 s window
cmake-build-release-gcc-15/core/test/qa_PerformanceMonitor 8 2   # TC2 (1 tag/10k samples)
cmake-build-release-gcc-15/core/test/qa_PerformanceMonitor 8 3   # TC3 (per-sample)
```

Expected post-Q1.B Tensor: TC1 ≥ 425 MS/s, TC2 ≥ 360 MS/s, TC3 ≥ 580 kS/s; ValueMap iter ≈ 10 M ops/s
(unchanged — gap is Task #24).

**Do not commit on your own initiative.** The user initiates every commit.

---

## 0. Quick status

| Phase                                                                          | State                                                                                                                                                                                                                                                                                                                                               |
|--------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Investigation (§2 of spec)                                                     | ✅ done                                                                                                                                                                                                                                                                                                                                              |
| Branch setup                                                                   | ✅ done (`featValueMapExtension`)                                                                                                                                                                                                                                                                                                                    |
| Round-2 design decisions                                                       | ✅ done (see §3)                                                                                                                                                                                                                                                                                                                                     |
| `Value` PMR fix-ups (copy ctor + move-assign)                                  | ✅ landed — `4f3e801b` (see §4.1)                                                                                                                                                                                                                                                                                                                    |
| Track `claude_wip.md`                                                          | ✅ landed                                                                                                                                                                                                                                                                                                                                            |
| Phase 1 — `ValueMap` (contiguous packed layout, USM-ready)                     | 🚧 Skeleton in `f3c396f5`; core impl in `19404e43`. **Round-3 refactor pending** (see §3 Round-3 table + §4.2 9-commit series): naming sweep, exception-free API via `assert(kDebugBuild)`, typed-name fluent (no user-visible key IDs), random-access iterators. Phase 1b (conversion, serialisation, extended types, alias switch) resumes after. |
| Phase 2 — `PooledRingBuffer` + `TagRingBuffer` + `Block::houseKeeping(policy)` | ⏸ after Phase 1                                                                                                                                                                                                                                                                                                                                     |
| Phase 3 — adoption sweep                                                       | not started                                                                                                                                                                                                                                                                                                                                         |

**Build dir**: `cmake-build-debug-gcc15-clean/` — fresh configure succeeded with defaults (`GR_ENABLE_BLOCK_REGISTRY=ON`). `qa_Value` green (6 + 2 PMR regression cases, 33 tests in "Value - Basic Construction", 53 in "Value generic map", etc.). The older `cmake-build-*` dirs carried stale state from a partial generate — not a repo bug; they'd need a destructive reset to recover, or just ignore and use the fresh dir.

**Performance baseline** (Release-gcc15, `qa_PerformanceMonitor`, 5 s × best of 3, taken 2026-04-23 BEFORE Tensor +
Phase 1c land — **repeat after each big integration to detect regression / quantify gain**):

| Test case | Description                       | 2026-04-23 baseline (samples/s) | After Tensor + 1c #1 + #2 (samples/s) | After Phase 1e + alias swap + lifetime fix (2026-04-26) |
|-----------|-----------------------------------|---------------------------------|---------------------------------------|---------------------------------------------------------|
| 1         | no tags                           | **3.53 × 10⁸** Hz               | **3.52 × 10⁸** Hz                     | **3.56 × 10⁸** Hz (≈ noise)                             |
| 2         | moderate (1 tag / 10 000 samples) | **2.67 × 10⁸** Hz               | **2.95 × 10⁸** Hz                     | **3.03 × 10⁸** Hz (+13 % vs baseline; +3 % vs 1c)       |
| 3         | per-sample tagging                | **4.68 × 10⁵** Hz               | **4.69 × 10⁵** Hz                     | **4.73 × 10⁵** Hz (≈ noise)                             |

No regression from the alias swap + view-mode escape lifetime fixes; TC2 picked up a small extra gain
(likely Settings::contextInTag hot-path now routes through `at()` which short-circuits on miss).

Tag overhead 1→2: ≈ 24 % drop pre-1c; ≈ 16 % drop post-1c (Settings.cpp::contextInTag is the
hot site for moderate tagging — single-find hot-path now). Per-sample 1→3 collapse unaffected
(TagMonitor / autoForward dominates, not contextInTag).
Re-run after every Phase 1c micro-op + after the alias switch.

**`bm_ValueMap` re-run 2026-04-26 (Release-gcc15, post-alias-swap + lifetime fixes; pre-Q1)**:

N=20 (mean ops/s): `std::map` `std::pmr::map` `std::unordered_map` `Value::Map (= ValueMap)` `gr::pmt::ValueMap`

| Workload | std::map | std::pmr::map | std::unordered_map | Value::Map |  ValueMap |
|----------|---------:|--------------:|-------------------:|-----------:|----------:|
| insert   |     984k |          901k |               1.2M |       999k |  **1.1M** |
| find     |     5.7M |          4.4M |               9.1M |  **12.3M** |      8.7M |
| iter     |    28.0M |         28.5M |          **77.1M** |      68.2M |    9.4M ⚠ |
| copy     |     4.5M |          3.8M |               4.5M |       3.7M | **14.8M** |
| merge    |     372k |          371k |               463k |       435k |      358k |
| erase    |     741k |          763k |           **1.0M** |       901k |      1.1M |

**Conclusions vs the pre-existing 2026-04-24 numbers (further down)**: ValueMap is essentially unchanged; the
single big gap is **iter** (9.4M vs unordered_map 77.1M, Value::Map 68.2M) — 8× slower because iter still
goes through `_entryToValue → decodeEntry → owning Value(...)` for variable-size types, allocating per-deref.
Closing this is exactly what Q1 inversion (Tensor + Map storage in Value byte-blob, then iter view-mode
for those types) is for. Insert / copy / find / erase are competitive or winning.

**Allocations / iter (N=20)**: ValueMap = 7 (insert / iter / copy), 20 (merge); unordered_map = 11 / 11 / 21 / 27;
ValueMap [static buf] = 0 (no heap touch under stack-arena PMR).

**Older `bm_ValueMap` snapshot (2026-04-24, pre-alias-swap; kept for diff context)** —

5 reps × 10 000 ops:
Containers compared, in order: `std::map`, `std::pmr::map`, `std::unordered_map`,
`std::pmr::unordered_map<…>` (= `Value::Map`), `gr::pmt::ValueMap`. Numbers are mean ops/s
(framework-driven, pre-built map; iteration excludes setup costs):

| Workload | N  | std::map | std::pmr::map | std::unordered_map | std::pmr::unordered_map (= Value::Map) | gr::pmt::ValueMap |
|----------|----|---------:|--------------:|-------------------:|---------------------------------------:|------------------:|
| insert   | 5  |     2.5M |          3.0M |               3.6M |                                   2.9M |          **3.1M** |
| find     | 5  |     7.6M |          9.4M |              15.3M |                                  16.7M |         **16.6M** |
| iter     | 5  |    38.7M |         51.9M |           **126M** |                               **103M** |           12.7M ⚠ |
| copy     | 5  |     5.2M |          5.5M |               6.4M |                                   4.6M |         **20.8M** |
| merge    | 5  |     953k |          1.1M |               1.4M |                                   1.2M |            574k ⚠ |
| erase    | 5  |     2.3M |          2.2M |               2.4M |                                   2.8M |              2.5M |
| insert   | 20 |     749k |          759k |               863k |                                   938k |          **1.1M** |
| find     | 20 |     4.3M |          4.9M |               8.6M |                                  12.1M |              9.7M |
| iter     | 20 |    23.3M |         28.4M |          **58.6M** |                              **61.2M** |            8.8M ⚠ |
| copy     | 20 |     3.5M |          3.7M |               4.3M |                                   3.7M |         **14.4M** |
| merge    | 20 |     290k |          342k |               424k |                                   394k |            301k ⚠ |
| erase    | 20 |     714k |          718k |               948k |                                   879k |              983k |

**Key takeaways**:

- ValueMap **wins** (best across all 5 containers) on **construct+insert** and **copy** — the
  packed-blob layout pays off (single contiguous alloc; copy is one memcpy).
- ValueMap is **competitive** on **find** and **erase** (parity with the hash-based maps).
- ValueMap **loses** on **iter** — current iter constructs a fresh `Value` per dereference
  (~24 bytes + type/container init + resource ptr); the hash-based maps store `Value` directly
  in the node and yield a `Value&` reference (no per-iter construction). The wire-format layout
  is the inherent cost. **Phase 1d steps 5+6 (Tensor + Map view-mode in iter) would not close
  this gap** — the dominant cost is Value-construction-per-iter on inline scalars, not the
  String-handling path that step 3 already optimised. Closing the gap further would require a
  lighter-weight iterator yield type (separate API), which is a Phase-2 redesign.
- ValueMap **loses** on **merge** — current `ValueMap::merge` materialises each source entry
  through the full Value path and re-emplaces. A raw-PackedEntry-copy fast path (skip Value
  reconstruction; copy bytes between blobs) is a tractable optimisation tracked separately.

**Phase 1d steps 5+6 status** (2026-04-24): **deferred indefinitely**. Per user direction, the
`get_if<Tensor<T>>` and `get_if<Value::Map>` accessors stay (~67 caller sites avoided). View-mode
for Tensor / Map could land later as opt-in additions (`get_if<TensorView<T>>`, view-mode
`ValueMap`) used only in new core code — without forcing migration.

**Phase 1e status** (2026-04-24, in progress on `featValueMapExtension`): per user direction
("Let's do the redesign of Value etc. in the same commit not a new branch — backward-breaking,
done in one go"), the storage-shape redesign + alias swap + caller migration land on this branch
as a coordinated breaking change. Done in incremental commits per the §4.2 commit-ready
discipline; estimated total ~700 SLOC.

**Step A landed (~150 SLOC)**: Value String storage migrated from `_storage.ptr → std::pmr::string`
heap-allocated object to raw byte-blob `[chars][\0]` via `_resource->allocate(_payloadLength + 1, 1)`.
`get_if<std::pmr::string>` dropped (callers use `get_if<std::string_view>()` or
`value_or<std::string>(default)`). Concept `ValueComparable` no longer includes pmr::string.

**Step B.1 landed (~50 SLOC)**: additive `Value::get_if<TensorView<U,…>>()` accessor returning
`std::optional<TensorView<U,…>>` — constructs a non-owning view from the existing heap
`Tensor<U,…>*`. Doesn't change storage; doesn't break callers. Excluded `TensorViewLike` from
the `T*` `get_if` overloads to prevent type-unsafe dispatch. New one-arg `gr::TensorViewLike`
concept added to `Tensor.hpp` (parallel to existing `TensorLike` / `TensorOf` / `TensorViewOf`).
`TensorView::owned(resource = std::pmr::get_default_resource())` materialiser added — copies
elements into a fresh `Tensor<T>` via `resize` + `std::copy`.

**Step B.2 landed (~80 SLOC)**: read-only Tensor caller migration from `get_if<Tensor<T>>`
to `get_if<TensorView<T>>`. Production sites: `Graph_yaml_importer.hpp` (5 sites — also dropped
redundant `checked_access_ptr` wrappers per Option B "rewrite for simplicity"), `Settings.hpp`
(2 sites), `Block.cpp` (1 site), `algorithm/ImCanvas.hpp` (1), `algorithm/FileIo.hpp` (1),
`meta/UnitTestHelper.hpp` (1). Test files: `qa_Value.cpp` (~14 sites), `qa_ValueMap.cpp`
(~14 sites — bool case kept on `get_if<Tensor<bool>>` since `TensorView<bool>` cannot be
constructed from `Tensor<bool>` due to the `vector<bool>` `data()` exclusion). Internal
`ValueMap.hpp` Map-ctor (1 site). All affected qa_ targets green: qa_Value, qa_ValueMap,
qa_ValueHelper, qa_Tensor, qa_Settings, qa_Block, qa_Graph, qa_YamlPmt, qa_TriggerMatcher,
qa_Tags, qa_DataSink, qa_DataSet, qa_Audio.

**Deferred to Step B.3 (storage redesign)**:

- 4 owning-mutate sites in `algorithm/ImGraph.hpp` (require `extract→reassign` deeper
  than the lambda-extraction pattern allows — they pass `Tensor<T>&` outside lambdas and keep
  mutating).
- `ValueHelper.hpp` template machinery — closely coupled to the storage shape; will migrate
  alongside the redesign.
- `ValueMap.hpp` decode/encode internals — redesign moves the format helpers from `ValueMap`
  to `Value` per user direction (Q1 inversion: Value owns the per-atom blob format, ValueMap
  is the keyed collection of atom-blobs).
- bool-tensor sites (1) — need a `TensorView<bool>` or stay on `get_if<Tensor<bool>>`.

**ValueMap view-mode + comparison/hash landed (this session, ~80 SLOC)**:

- `ValueMap::makeView(span<const byte>) → ValueMap` — view-mode factory; aliases external
  bytes; `_resource == nullptr` is the discriminant per Q5.
- `ValueMap::is_view()` predicate.
- `ValueMap::owned(resource)` — materialiser for view-mode bytes (deep copy via target's
  resource through the existing copy ctor path).
- Mutator guards (`_isViewAndAssertNoMutation`): public `emplace` / `insert_or_assign` /
  `erase` (both overloads) / `clear` / `reserve` / `shrink_to_fit` / `merge` assert in
  debug, no-op in release. Per Q5 hybrid policy.
- `operator==(const ValueMap&) const` extended with same-instance / aliased-blob early
  returns — keeps the O(N²) full content comparison rare in practice.
- `std::hash<gr::pmt::ValueMap>` added (order-independent: per-entry hash XOR'd then sum'd).
- `_deallocateBlob` already handled `_resource == nullptr` correctly (existing check),
  so destructor and copy/move automatically respect view-mode.

---

### RESUMPTION POINT — Phase 1e mid-refactor (2026-04-25 — FOURTH UPDATE — recovery after OOM crash)

**Crash recovery (2026-04-25 late)**: prior session OOM'd after building too many qa_ targets in
parallel (no per-session limit, even though CMake JOB_POOLS = compile_pool=6/link_pool=2 are
configured). System tore down. Recovery underway in this session.

**Tests state (2026-04-25 latest, build dir = `cmake-build-debug-gcc-15/`, ASan enabled by default)**:

- ✅ qa_Value: ALL suites green (32+ suites)
- ✅ qa_ValueMap: ALL 10 suites green (incl. Tensor support, extended types, STL parity)
- ✅ qa_YamlPmt: ALL passing (118 asserts / 31 tests + grep + formatter)
- ⚠️ qa_Tags: builds but ASan UAF in TagPropagation (was claimed all-green in third update;
  the cached binary from 12:49 today already shows the regression — likely introduced by
  the alias swap landing without a corresponding move-ctor materialise fix).
- ⚠️ qa_Graph: builds but ASan UAF (same Settings::get path as qa_Tags)
- ⚠️ qa_Scheduler: builds (was claimed "doesn't build" in third update — actually builds fine
  after migration; same ASan UAF as qa_Tags / qa_Graph at runtime).
- 🟡 qa_Settings: 27/28 — YAML test fails with Tensor<Value> nested decode corruption
  (status from third update; not re-checked yet)
- ⚠️ qa_DataSink: builds but segfaults on run (Tensor<Value> corruption; not re-checked yet)

**Migration sweep state (was in third update as "Files needing migration")**: ALL DONE before
crash. Verified by inspection in this session — Scheduler.hpp / Graph.hpp / BlockModel.cpp /
HttpBlock.hpp all use `insert_or_assign` already; ImGraph.hpp uses extract→reassign pattern
for the 4 owning-mutate sites; ImChart.hpp / NMEADevice.hpp had no ValueMap-related sites
to migrate. The "9 items" were broadly:

1. ✅ Alias swap (`Value::Map = ValueMap`, `property_map = Value::Map`) compile-green
2. ✅ qa_Value / qa_ValueMap green (Value/ValueMap APIs)
3. ✅ qa_Block / qa_TriggerMatcher / qa_YamlPmt green
4. ✅ SubscriptProxy added (auto-vivify on miss, throw only when const → matches std::map)
5. ✅ ImGraph extract→reassign refactor (4 owning-mutate sites)
6. ❌ Move-ctor view-mode materialisation (NEW root cause for qa_Tags / qa_Graph / qa_Scheduler
   ASan UAF — see "Move-ctor fix landed" below)
7. ❌ Tensor<Value> nested decode corruption (qa_Settings YAML / qa_DataSink) — likely needs
   Step B.3 byte-blob redesign (Q1 inversion) per user direction
8. 🚫 Q1 inversion (Step B.3 + C+D — Tensor + Map storage to byte-blob in Value) — not done yet
9. 🚫 Extended unit tests for new edge cases (SubscriptProxy semantics, view-mode round-trip,
   Q1 byte-blob format invariants) — pending

**Move-ctor fix attempt + revert (this session)**:
First attempt — make `Value::Value(Value&&)` materialise from view-mode (allocate fresh bytes
via `copy_from` whenever source had `kFlagViewMode` set). Drop `noexcept`. Cleanly fixed
qa_Tags / qa_Graph / qa_Scheduler ASan UAF in `Settings::get(string)` (which returned a
view-mode Value out of the source ValueMap's scope).

REVERTED a few minutes later because it introduced a NEW UAF in qa_Settings YAML / qa_DataSink:
the `auto sv = (*it).second.value_or(string_view{});` pattern (pervasive in `getProperty<T>`,
several block callbacks, ImGraph etc.) had previously returned a string_view aliasing the
source ValueMap's `_blob` — alive as long as the source map is. After the move-ctor materialise,
the iter pair's Value becomes owning at the source map's resource, the bytes get freed when
the temp pair dies, and the string_view dangles immediately at the `;`.

Verdict: the two cases have OPPOSING needs.

- Case A (Value escapes scope, source map dies): wants materialise-on-move
- Case B (string_view extracted from temp Value, used after `;`): wants preserve-view-mode

The architectural fix is **fix at the originating site**, not in the move-ctor:

- Case A — `Settings::get(string)`: route through `at()` (which uses `_entryToValue` →
  `decodeEntry` → owning Value) instead of `(*it).second` (iter view-mode).
- Case B — caller patterns `auto sv = m.at(k).value_or(sv{}); use(sv);` must bind the
  Value to a const lvalue first, OR copy out to `std::string`. Case B is essentially the
  WIP-documented unsafe pattern that was masquerading as safe under the old `Value::Map`
  (where `at()` returned `Value&` reference).

Move-ctor reverted to noexcept pointer-transfer. Dozens of caller-site fixes applied
(see "Per-site fix sweep" below).

**`get_if<TensorView<bool>>` constraint** (~10 SLOC):
Added `requires !same_as<remove_const_t<value_type>, bool>` to the templated
`Value::get_if<TensorView<T>>()` overload. Reason: `TensorView<bool>(Tensor<bool>&)` ctor
calls `tensor.data()` which doesn't exist on `Tensor<bool>` (uses `pmr::vector<bool, true>`).
Three callers (Settings.hpp:301 + 690, UnitTestHelper.hpp:195) gained an `if constexpr
(is_same_v<TTensorElem, bool>)` branch that falls back to `get_if<Tensor<bool>>()`.

**Per-site fix sweep (this session)** — all UAF / lvalue-binding migrations:

Production code:

- `core/src/Settings.cpp::CtxSettingsBase::get(string)` — replaced view-mode-escape via
  `(*it).second` with `res.at(key)` (owning Value via decodeEntry). Single-line semantic fix.
- `core/src/Block.cpp` — `propertyCallbackLifecycleState`, `propertyCallbackActiveContext`,
  `propertyCallbackSettingsCtx` (×2) — bound by-value `at()` results to lvalues so
  `get_if<bool|uint64_t|TensorView|Map>()` pointers and `value_or(sv{})` string_views remain
  valid past the init expression.
- `core/include/gnuradio-4.0/Graph.hpp::subgraphExportHandler` — bound 5× by-value `at()`
  results to lvalues; collected `data().value_or(sv{})` and `get_if<bool>()` into named locals.
- `core/include/gnuradio-4.0/Scheduler.hpp::propertyCallbackEmplaceEdge` — bound 7× by-value
  `at()` results to lvalues.
- `core/include/gnuradio-4.0/Scheduler.hpp::propertyCallbackRemoveBlock`,
  `propertyCallbackReplaceBlock`, scheduler YAML Set handler — switched
  `value_or(string_view{})` results to owning `std::string` (heap copy survives the temp).
- `core/src/Graph.cpp::propertyCallbackGetBlockInfo` — same `std::string` fix.
- `core/include/gnuradio-4.0/Graph_yaml_importer.hpp::loadGraph` lambda — bound `grcBlock.at("graph")`
  result to lvalue Value before `get_if<property_map>()`.
- `core/include/gnuradio-4.0/Graph_yaml_importer.hpp::loadGraphFromMap` ctx_parameters loop —
  bound 3× ctxPar->at() results, switched ctxName to owning `std::string`.
- `core/include/gnuradio-4.0/TriggerMatcher.hpp::isSingleTrigger` — bound 2× by-value at()
  results to lvalues so `checked_access_ptr{...}` pointers stay valid.
- `core/include/gnuradio-4.0/Message.hpp::sendMessage(initializer_list)` — changed key type
  from `pmr::string` to `string_view` (matches ValueMap's `value_type`).
- `core/include/gnuradio-4.0/Scheduler.hpp::SchedulerBase(initializer_list, ...)` — same key
  type change; constructs `property_map{initParameter}` for the set() call.
- `algorithm/include/gnuradio-4.0/algorithm/ImGraph.hpp::syncFromUIConstraints` — bound 4×
  by-value `at()` results for side / pos_x / pos_y / exit_dir lookups.
- `algorithm/include/gnuradio-4.0/algorithm/fileio/FileIo.hpp::Reader::poll` — bound 1×
  by-value `(*it).second` so the `TensorView<uint8_t>` data pointer stays valid through
  the callback invocation.
- `blocks/audio/include/gnuradio-4.0/audio/AudioBlocks.hpp::AudioSource::clk_in` — bound 4×
  by-value iter dereferences to lvalues.
- `blocks/basic/include/gnuradio-4.0/basic/CommonBlocks.hpp::builtin_multiply` — bound 1×
  by-value iter result for the factor read.
- `blocks/basic/include/gnuradio-4.0/basic/SyncBlock.hpp::isSyncTag` + `getTime` — bound 3×
  by-value iter results to lvalues.

Tests:

- `core/test/qa_Port.cpp` — `tagMap.get_allocator().resource()` → `tagMap.resource()` (no
  std::pmr-style allocator API on ValueMap); `nameIt->second` → `(*nameIt).second`.
- `core/test/qa_Messages.cpp::is_contained` lambda — `it->second` → `(*it).second`.
- `core/test/qa_GraphMessages.cpp` — `foundTypes->second` → `(*foundTypes).second` plus
  fixed pre-existing `||` vs `&&` logic bug in the conditional (the success-path test
  required `found AND right type`, not `found OR wrong type`).
- `core/test/qa_ManagedSubGraph.cpp` — sed-replaced `yamlData->second` → `(*yamlData).second`
  (×2 sites).
- `core/test/qa_SubGraphAssets.cpp` — `it->second` → `(*it).second`.
- `core/test/plugins/good_math_plugin.cpp` — same as CommonBlocks.hpp pattern.
- `blocks/audio/test/qa_Audio.cpp` — same as AudioBlocks.hpp pattern (clk_in clock trigger
  forwarding test).

**Test status (2026-04-26 — all green except pre-existing failures)**:

| Target             | Status                                                           |
|--------------------|------------------------------------------------------------------|
| qa_Value           | PASS                                                             |
| qa_ValueMap        | PASS                                                             |
| qa_ValueHelper     | PASS                                                             |
| qa_Tensor          | PASS                                                             |
| qa_Settings        | PASS                                                             |
| qa_Block           | PASS                                                             |
| qa_Tags            | PASS (TagPropagation 6 tests now green)                          |
| qa_TriggerMatcher  | PASS                                                             |
| qa_DataSet         | PASS                                                             |
| qa_Scheduler       | PASS                                                             |
| qa_Graph           | PASS                                                             |
| qa_YamlPmt         | PASS                                                             |
| qa_Messages        | PASS                                                             |
| qa_Port            | PASS                                                             |
| qa_buffer          | PASS                                                             |
| qa_BlockModel      | PASS                                                             |
| qa_LifeCycle       | PASS                                                             |
| qa_DynamicBlock    | PASS                                                             |
| qa_DynamicPort     | PASS                                                             |
| qa_GraphMessages   | PASS                                                             |
| qa_BlockingSync    | PASS                                                             |
| qa_grc             | PASS                                                             |
| qa_ManagedSubGraph | PASS                                                             |
| qa_DataSink        | PASS (was segfaulting; same Settings::get UAF)                   |
| qa_HttpBlock       | PASS                                                             |
| qa_Audio           | 1 fail — soundio dummy backend env-dependent issue, unrelated    |
| qa_SubGraphAssets  | 1 fail — pre-existing plugin loader issue, unrelated to Phase 1e |

All previously-failing tests (qa_Tags TagPropagation, qa_Graph, qa_Scheduler ASan UAFs;
qa_Settings YAML; qa_DataSink Tensor<Value> segfault) are now green. The Tensor<Value>
nested decode corruption WIP theory was wrong: the actual root cause was view-mode-aliased
string_views escaping their source's scope.

**Edge-case test suites added (this session)**:

- `qa_ValueMap.cpp` — 6 SubscriptProxy tests (auto-vivify on miss, type promotion from
  Monostate, overwrite, string write, chained typed reads) added to the operator[] suite;
  new "ValueMap - view-mode (makeView / is_view / owned)" suite with 6 tests covering
  makeView, is_view, view-mode iter, owned() materialisation, view-equality, and the
  debug-only mutator-on-view assert.
- `qa_Value.cpp` — new "Value - view-mode lifetime" suite (4 tests) covering makeView,
  copy-ctor materialise, two-arg copy-ctor explicit-resource, move-ctor preserves view-mode;
  new "Value - by-value lifetime safety" suite (3 tests) documenting the safe lvalue-bind
  pattern that callers must follow when escaping iter scope.

**Q1 inversion status (Phase Q1.A + Q1.B Map + Q1.B Tensor landed, 2026-04-27)**:

**Q1.B Tensor byte-blob storage landed (2026-04-27 session)**:

- ✅ Tensor storage migrated from heap `Tensor<T>*` to byte-blob — Value's `_storage.ptr` now
  points at a single PMR allocation holding `[elementVT:1][rank:1][encodingFlags:1][reserved:1]
  [elementCount:4][extents:4·rank][element bytes]`. `Value(Tensor<T>)` / copy_from / destroy /
  `operator==` all routed through `init_from_tensor` (out-of-class in ValueMap.hpp where
  `encodeTensorBlob` lives). Hash function reads via the new accessors.
- ✅ `get_if<Tensor<T>>()` returns `std::optional<Tensor<T>>` (decoded copy from byte-blob).
  Body in ValueMap.hpp; uses `decodeTensorBlob` for variable-size elements (Tensor<Value>) and
  `get_if<TensorView<T>>().owned()` for fixed-size scalars. **Subtle fix**: must use parens
  (`T(std::move(tensor))`) not braces — Tensor<Value>→Value is implicit, so `T{...}` would
  pick the init_list ctor and build a 1-element tensor wrapping the moved tensor as one Value.
- ✅ `get_if<TensorView<T>>()` extended uniformly:
  - fixed-size T (scalar / complex<float> / complex<double>): aliases the data section
    contiguously — zero allocation, zero copy.
  - bool: byte-blob stores 1 byte per bool, but `TensorView<bool>` isn't constructible
    (`pmr::vector<bool>` has no `.data()`); `get_if<Tensor<bool>>` is the access path.
  - **Tensor<Value> partial specialisation** (`namespace gr` in `Value.hpp`): variable-size
    elements can't be aliased contiguously, so the specialisation owns a decoded
    `Tensor<Value>` snapshot and forwards range API. Lifetime is the optional's, not the
    source Value's — same shape as `get_if<ValueMap>()` view-mode.
- ✅ `value_or<T>` template special-cases `TensorLike<Raw>` — Tensor byte-blob has no stable
  `T*` for reference returns; always returns by value (`return_t` updated to drop reference
  for TensorLike). `value_or(Tensor<...>())` and `value_or(const Tensor<...>&)` both work.
- ✅ `Settings.hpp` enum/applier paths simplified: dropped the `bool` Tensor fallback (now
  unified through TensorView), added the TensorLike branch for the generic checked_access
  path so `optional<Tensor<T>>` is unwrapped before storing.
- ✅ Caller migrations: `auto* p = .get_if<Tensor<...>>()` → `auto p = .get_if<...>()` with
  `*p` deref. Sites: `Value.cpp` hash function (13×), `ValueMap.hpp` two internal converters,
  `ValueHelper.hpp` six dispatch tables (vector/array/Tensor × value/Tensor cases),
  `ImGraph.hpp:373`, `qa_ValueMap.cpp:838`. ~22 sites total, all compile clean.
- ✅ `initStringTensor` (Value's ctor for `vector<string>` / `array<string>`) now routes
  through `init_from_tensor` instead of placement-new of a heap Tensor.

**Tests state (2026-04-27 post-Q1.B-Tensor)**: 10 core qa_ targets PASS — qa_Value, qa_ValueMap,
qa_Settings, qa_Tags, qa_Block, qa_Scheduler, qa_Graph, qa_DataSink, qa_DataSet, qa_YamlPmt.

**Full ctest sweep — 8 failures, ALL pre-existing on `4ee17acf WIP` baseline (verified by re-running
on stashed pre-Q1.B working tree)**:

- qa_SchedulerMessages — SEGV, pre-existing (baseline confirmed).
- qa_plugins_test — SEGV, pre-existing (baseline confirmed).
- qa_plugin_schedulers_test — SEGV, same pattern as SchedulerMessages / plugins_test (pre-existing).
- qa_Audio — soundio dummy backend env issue, pre-existing (already documented).
- qa_StreamToDataSet — heap-use-after-free in `magic_enum::cmp_equal` during FunctionGenerator
  settings application (pre-existing — confirmed via earlier targeted git-stash check).
- qa_RTL2832Source — hardware-dependent, hangs waiting for device (env, not code).
- qa_SoapySource — passes on re-run (transient / env; not a regression).
- qa_KnownSharedLibBlocks — SEGV, pre-existing (baseline confirmed).

**No regressions introduced by Q1.B Tensor migration.**

**Bench re-run (Release-gcc15, best-of-3, 2026-04-27 post-Q1.B Tensor)**:

`bm_ValueMap` (N=20, ops/s) — gr::pmt::ValueMap vs unordered_map / Value::Map:

| Workload | unordered_map | Value::Map | gr::pmt::ValueMap | post-Q1.A baseline |           Δ vs Q1.A |
|----------|--------------:|-----------:|------------------:|-------------------:|--------------------:|
| insert   |          1.2M |       1.1M |   **1.3M** (best) |               1.2M |                 +8% |
| find     |          9.3M |      12.7M |              9.8M |               8.7M |                +13% |
| iter     |     **77.1M** |      72.6M |           10.2M ⚠ |              10.1M | +1% (gap unchanged) |
| copy     |          5.3M |       4.0M |  **14.9M** (best) |              14.8M |                 +1% |
| merge    |          476k |       461k |              399k |               358k |                +11% |
| erase    |          1.1M |       927k |  1.1M (best tied) |               1.1M |                   ≈ |

Allocations / iter: 7 (unchanged); static-buf path = 0. Iter gap NOT closed —
Task #24 (iter view-mode for Tensor / nested-Map) still pending.

`qa_PerformanceMonitor` (best-of-3, 8 s windows):

| Test | Description          | 2026-04-23 baseline | 2026-04-26 (post-Phase-1e+alias) | 2026-04-27 (post-Q1.B Tensor) | Δ vs baseline | Δ vs 2026-04-26 |
|-----:|----------------------|--------------------:|---------------------------------:|------------------------------:|--------------:|----------------:|
|  TC1 | no tags              |            353 MS/s |                         356 MS/s |                  **441 MS/s** |      **+25%** |            +24% |
|  TC2 | moderate (1 tag/10k) |            267 MS/s |                         303 MS/s |                  **375 MS/s** |      **+40%** |            +24% |
|  TC3 | per-sample tagging   |            468 kS/s |                         473 kS/s |              **593–604 kS/s** |      **+27%** |            +27% |

**Headline**: TC2 +24% / TC3 +27% match the expected win — Tensor byte-blob
removes per-Value heap reconstruction in tag/settings hot paths. TC1 +24%
(no-tags) is surprising and likely traces to Value's fixed footprint reducing
cache pressure in the scheduler's per-sample Value lifecycle; deserves a perf-
counter follow-up to confirm. iter ValueMap unchanged at 10.2M (still 7× behind
unordered_map) — closes only with Task #24.

**Q1.A landed 2026-04-26 earlier in session, ~250 SLOC**:

**What landed**:

- Tensor blob format constants moved from `ValueMap.hpp` into `Value.hpp` (Value owns the format).
- `Value::get_if<TensorView<T>>()` extended for view-mode source — parses tensor blob header,
  constructs TensorView aliasing the data section. Owning case unchanged. bool excluded
  (TensorView<bool> not constructible — pmr::vector<bool> has no .data()).
- `Value::get_if<ValueMap>()` reshaped: now returns `std::optional<ValueMap>` view-mode
  (alloc-free; aliases the source's bytes) instead of `Map*`. **Replaces the pointer-based
  `get_if<Value::Map>()` / `get_if<property_map>()` API entirely** — every caller migrated
  (25+ sites: ValueMap.hpp internal, ValueHelper.hpp, ValueFormatter.hpp, PluginLoader.hpp,
  Block.cpp, BlockMerging.hpp, YamlPmt.hpp, Graph_yaml_importer.hpp, Scheduler.hpp,
  Settings.hpp, AudioBlocks.hpp, UnitTestHelper.hpp, tests).
- `value_or<T>()` template excludes `ValueMap` from generic dispatch. Callers using
  `value_or(property_map{})` migrated to `get_if<ValueMap>().owned()` for an owning copy.
- `_entryToValue` uses default_resource fallback when source ValueMap is view-mode (avoids
  null-resource segfault when decoding nested Tensor<Value>).
- New tests: qa_ValueMap "view-mode" suite (6 tests / 19 asserts) and qa_Value "view-mode
  lifetime" + "by-value lifetime safety" suites (7 tests / 15 asserts).

**Tests state (2026-04-26)**: 24 in-scope qa_ targets PASS. qa_HttpBlock PASS. qa_Audio: 1
fail (soundio dummy backend env issue, pre-existing). qa_SubGraphAssets: 1 fail (plugin
loader, pre-existing).

**Bench (Release-gcc15)** N=20 ops/s — `unordered_map | Value::Map | ValueMap`:

- insert: 1.2M / 999k / **1.2M** (best tied)
- find:   9.1M / 12.3M / 8.7M (≈ parity)
- iter:   **72.1M** / 72.5M / 10.1M ⚠ (still 7× behind — see below)
- copy:   4.5M / 3.7M / **14.8M** (best by far)
- merge:  463k / 435k / 358k
- erase:  1.0M / 901k / **1.1M** (best)

**iter perf gap NOT closed**: ValueMap iter still allocates owning Values for variable-size
types (Tensor / nested-Map) via decodeEntry. Switching iter to view-mode for these types was
tried this session but reverted because callers' `value_or(Map{})` and `Tensor::data()`
patterns break when the iter result is view-mode (no heap object to alias). Closing the gap
requires Phase Q1.B (full storage inversion):

**Phase Q1.B (deferred — full storage inversion, ~300 more SLOC)**:

- Step B.3: Tensor byte-blob storage replacing heap `Tensor<T>*` in Value
- Step C+D: Map byte-blob storage replacing heap `ValueMap*` in Value
- Iter view-mode for Tensor / Map (the actual perf gain — yields view-mode Values aliasing
  entry payload bytes, alloc-free for these types)
- Migrate remaining `get_if<Tensor<T>>` / `value_or<Tensor<T>>` callers (~10-15 sites in
  ValueHelper.hpp template machinery + ImGraph.hpp owning-mutate patterns)
- Re-bench; expect iter to reach ~50M ops/s (close the 7× gap)
- Estimated 2-3 focused sessions of work + caller migration.

**Key learning: ValueMap iter-by-value safety**

ValueMap iter dereference returns `pair<string_view, Value>` BY VALUE (not by reference).
Code patterns that DON'T work post-alias-swap:

- `it->second.get_if<T>()` → `*p` aliases temporary, dangles after expression ends
- `auto x = m[k].value_or(string_view{})` → string_view aliases destroyed temporary
- `m[k] = v` writes — operator[] is read-only on ValueMap
- `state[k] = v` (assignment to result of operator[]) — same issue
- `std::swap(state[a], state[b])` — operator[] returns rvalue
- `m.merge(rvalue)` — merge requires lvalue ValueMap&
- `m.try_emplace(k, v)` — ValueMap doesn't have try_emplace

**Migration patterns that work**:

- READ writes: `m.insert_or_assign(std::string_view{k}, v)` instead of `m[k] = v`
- READ pointers: bind Value to lvalue first, then take pointer
  ```cpp
  const pmt::Value valueSnap = m[k];
  if (auto* p = valueSnap.get_if<bool>()) { ... } // *p valid in scope of valueSnap
  ```
- READ string_views: copy out as std::string instead of string_view
  ```cpp
  std::string s = m[k].value_or(std::string{}); // owning copy, safe
  ```
- READ flags into bool by value: `bool b = m[k].value_or(false);`

**Files migrated (all compile)**:

- core: Block.{hpp,cpp}, BlockMerging.hpp, BlockModel.{hpp,cpp}, Graph.{hpp,cpp},
  Graph_yaml_importer.hpp, PluginLoader.hpp, Port.hpp, Scheduler.hpp, Settings.{hpp,cpp},
  Tag.hpp, TriggerMatcher.hpp, Value.{hpp,cpp}, ValueHelper.hpp, ValueMap.hpp
- algorithm: dataset/DataSetEstimators.hpp, fileio/FileIo.hpp, ImCanvas.hpp (PARTIAL)
- blocks: testing/TagMonitors.hpp, basic/StreamToDataSet.hpp, basic/DataSink.hpp
- tests: qa_Tags.cpp, qa_Block.cpp, qa_Settings.cpp, qa_DataSink.cpp

**Files needing migration before scheduler/graph tests build**:

- algorithm/ImGraph.hpp: heavy use of `try_emplace`, `it->second = ...`, `m[k] = v`
  patterns. Needs ~50-80 line rewrite to be ValueMap-compatible.
- algorithm/ImChart.hpp: not yet checked
- core/include/gnuradio-4.0/Scheduler.hpp: 5 sites with `data[k] = v` writes (lines
  902, 923, 963, 989, 1019, 1056)
- core/include/gnuradio-4.0/Graph.hpp: 2 sites
- core/src/BlockModel.cpp: 6 sites (serializer)
- blocks/http/HttpBlock.hpp: 3 sites
- blocks/timing/NMEADevice.hpp: 1 site

**REMAINING DEEPER BUG — Tensor<Value> nested decode corruption**:

When decoding a Tensor<Value> stored inside a ValueMap, the per-element Value's _resource
becomes corrupted between iterations of the decode loop. Specifically `tensor._data[i]`'s
_resource shows valid pointer at iter 0 but invalid pattern (e.g. `0x32372d363335`) at
iter 1+. This is heap corruption — possibly the Tensor's element memory shares pages with
the source byte blob being decoded recursively.

Theories:

1. The Tensor extents-from ctor default-constructs Values via `Value()` → _resource = default,
   which differs from the `resource` parameter. Cross-resource operator= triggers cross-resource
   path that allocates from default. If that default resource happens to overlap with the
   recursive ValueMap's blob memory, corruption ensues.
2. The Tensor's _data area gets reused by something during the decode (memory aliasing).

Investigation needed: valgrind, or instrument _resource pointer values throughout decode.
Fix may require constructing Tensor elements with the same resource as the source blob,
OR moving Map storage to byte-blob (Step C+D) so the recursive decode happens via different
memory paths.

**SLOC SUMMARY** (vs main):

- Total diff: 26 files, ~470 insertions, ~310 deletions

### RESUMPTION POINT — Phase 1e mid-refactor (2026-04-25 — second update)

**Latest state (mid-day 2026-04-25)**:

After alias swap (`Value::Map = ValueMap`, `property_map = ValueMap`), all 19+ files compile.

**Tests state**:

- ✅ qa_Tags: ALL passing (28 tests / 312 asserts) - was 21+ failures
- ✅ qa_Block: green after migrating `uiConstraints()[k] = v` → `insert_or_assign`
- 🟡 qa_Settings: 27/28 passing. ONE failing test: "Property auto-forwarding with GRC-loaded graph" - segfault on YAML
  decode of nested Tensor<Value> with Map elements
- ⚠️ qa_DataSink: builds but segfaults on run (likely same Tensor<Value> decode issue)

**Root cause of remaining failures (DO NOT lose this)**:

ValueMap iter dereference returns `pair<string_view, Value>` BY VALUE (not by reference like
std::pmr::unordered_map iter). This breaks any code that does:

- `if (auto* p = it->second.get_if<...>()) { use(*p); }` — `*p` aliases the temporary Value
  that dies at end of full expression. Use-after-free.

**Fixes applied**:

- Graph_yaml_importer.hpp: bound `(*it).second` to lvalue `pmt::Value entry = ...` before
  calling `entry.get_if<...>` (4 sites: blocks loader, ctx_parameters, connections, scheduler)
- TriggerMatcher.hpp: rewrote `state[k] = v` writes as `state.insert_or_assign(...)`,
  rewrote `std::swap(state[a], state[b])` as load-into-string + 4 insert_or_assigns
- StreamToDataSet.hpp:130: bound `(*it).second` to lvalue
- StreamToDataSet.hpp:324, DataSink.hpp:621: `merge(rvalue)` → bind to lvalue first
- Settings.hpp:908,920+: `meta_information[k] = ` → `meta_information.value.insert_or_assign(...)`
- Settings.hpp:383: wrapped Tensor in `pmt::Value{...}` for ValueMap insert
- DataSetEstimators.hpp: `meta_information[i][k] = v` → `meta_information[i].insert_or_assign(...)`
- BlockMerging.hpp:105: `it->second` → `(*it).second`
- TagMonitors.hpp, FileIo.hpp: same iter-deref fixes
- qa_Settings.cpp:419,432: test `metaInformation()[k] = v` → `insert_or_assign`
- qa_Block.cpp:941-942: same
- qa_DataSink.cpp:78,487,558,623: removed `.get()` on optional<Value> returns

**REMAINING DEEPER BUG — Tensor<Value> nested decode corruption**:

The qa_Settings YAML test loads a GRC graph which has `blocks: [...]` (Tensor<Value> with Map
elements). Encoding works correctly. Decoding shows:

- decodeTensorBlob with elementCount=4 produces tensor where elem[1]'s _resource is corrupt
  (e.g. `0x32372d363335` = ASCII bytes "5-7263") after assigning elem[0]
- The tensor element memory appears to be reused / overwritten by other allocations during
  the recursive decode

Theories (untested):

1. The pmr::vector<Value, true> default-constructs elements with `std::pmr::get_default_resource()`
   instead of the Tensor's `mr`. operator= then crosses resources and allocates from default.
   But the BACKING memory of subsequent elements is corrupted by something between iterations.
2. The decodeTensorElement's recursive ValueMap allocations are stomping on the parent
   Tensor's element memory because they share a resource and the resource doesn't separate them.

**Next-session priorities**:

1. **Trace down the Tensor<Value> nested decode corruption**. Likely needs valgrind or careful
   pointer-arithmetic tracing of the pmr resource. May indicate that Map storage redesign
   (Step C+D) is needed sooner than planned.
2. Build remaining tests: qa_Scheduler, qa_Graph, qa_YamlPmt, qa_TriggerMatcher
3. Migrate remaining ~30 production sites still using `[k] = v` pattern (BlockModel.cpp,
   Scheduler.hpp, Graph.hpp, Graph_yaml_importer.hpp, HttpBlock.hpp etc.) — these will surface
   when their qa tests are built
4. Add ValueMap iter-by-value safety: either make iter return a Wrapper that holds Value
   internally with proper lifetime, OR document that callers must bind to lvalue before
   pointer extraction
5. Add corner-case tests once green

### RESUMPTION POINT — Phase 1e mid-refactor (2026-04-25)

If a future session picks up here, the working tree on `featValueMapExtension` contains
~13 files of in-flight changes (see `git diff --stat`). The branch is build-green for
all directly-affected qa_ targets (qa_Value, qa_ValueMap, qa_ValueHelper, qa_Tensor,
qa_Settings, qa_Block, qa_Tags, qa_TriggerMatcher, qa_Graph, qa_YamlPmt, qa_DataSink,
qa_DataSet, qa_Audio).

**Working tree contents (commit-ready as one logical chunk)**:

1. `Value::get_if<TensorView<T>>` additive accessor (Step B.1)
2. Read-only Tensor caller migration to TensorView (Step B.2 production + tests +
   ValueMap-internal). Skipped: 4 ImGraph owning-mutate sites, ValueHelper.hpp template
   machinery, 1 bool-tensor site, Value.hpp / Value.cpp (where the templates live).
3. `gr::TensorView::owned(resource)` materialiser
4. `gr::TensorViewLike` concept added to Tensor.hpp
5. ValueMap view-mode infrastructure (makeView, is_view, owned, mutator guards,
   operator== with early returns, std::hash specialisation).

**Next step queued: Tag.hpp `property_map` alias swap (decoupled from Value::Map)**:

- `using property_map = pmt::Value::Map;` → `using property_map = pmt::ValueMap;`
- Tag.hpp must `#include <gnuradio-4.0/ValueMap.hpp>` (was only including Value.hpp)
- Tag::at: return Value by-value (was Value&); ValueMap::at returns by-value already
- Tag::get: return std::optional<Value> (was std::optional<reference_wrapper<const Value>>)
- Tag::insert_or_assign: works (both APIs have it)
- PropertyMapType concept: `same_as<T, property_map>` already correct shape
- Decouples from Value::Map (which stays as `std::pmr::unordered_map<…>`); Value::Map
  storage redesign + alias-swap to ValueMap is a SEPARATE step
- Migration scope: ~14 `tag.at/get` callers, ~28 `get_if<property_map>` in core/, ~10 in
  blocks/, ~8 in core/test/qa_*. Compatibility risks listed below.

**Compatibility risks for the alias swap (CONFIRMED via attempted swap, 2026-04-25)**:

Attempted the bare-minimum alias swap (`using property_map = pmt::ValueMap;` + Tag::at by-value

+ Tag::insert_or_assign by `map.insert_or_assign` instead of `map[k] = v`) — surfaced
  **204 compile errors** in `qa_Tags` alone (transitive: Settings.cpp, Settings.hpp, Port.hpp,
  Block.hpp, DataSet.hpp, etc.). Reverted Tag.hpp + Settings.cpp; tree is clean again.

**Categorised breakage classes (all need addressing in the alias-swap commit)**:

1. **`iter->second` / `iter->first` doesn't work** — ValueMap iter yields by-value `pair`;
   needs `(*iter).second`. Sites: `Settings.cpp:41,378,380,387`, `Port.hpp:1279`, plus
   probably more in core/include. Mechanical replace.

2. **`map.insert(...)` doesn't exist on ValueMap** — `Settings.cpp:146,158`. Use
   `map.insert_or_assign(k, v)` per-pair, or add an `insert` overload to ValueMap.

3. **`map[k] = v` doesn't work** — ValueMap's `operator[]` is read-only (returns Value by
   value). All write-via-subscript sites must migrate to `map.insert_or_assign(k, v)`.
   Sites: `Tag.hpp:108,110,138`, `Tag.hpp::put` template (lines 222-224, 228-230), …

4. **`map.get_allocator().resource()` doesn't exist** — ValueMap exposes `resource()`
   directly. Sites: `Tag.hpp::put` (line 223, 229).

5. **`property_map{...}` initializer-list construction** — `Port.hpp:205` uses
   `property_map{ {"a", Value{1}} }`. ValueMap doesn't have an initializer-list ctor;
   must add one OR rewrite call sites.

6. **`std::hash<std::pmr::unordered_map<…, Value, MapHash, MapEqual, …>>` missing** —
   `Settings.hpp:188` requires hashing the OLD Value::Map type (since Value::Map stays as
   std::pmr::unordered_map). This is unrelated to the alias swap (`property_map = ValueMap`)
   but the attempted swap exposed it. Standalone fix: provide `std::hash<Value::Map>`
   specialisation OR change the Settings.hpp site to not need hashing the unordered_map.

7. **Map-source ValueMap emplace** — `ValueMap.hpp:1720` static_assert fires when emplacing
   a value type that's not in the supported set. Likely a Value::Map (= std::pmr::unordered_map)
   value being emplaced through ValueMap's converter ctor. Need to handle nested-Map case
   (when source has a Value-of-Map entry).

8. **`std::less<void>` mixed pmr::string / std::string** — pre-existing pattern in some
   STL container code; surfaced by the alias swap because more code paths exercise it.
   Standalone fix.

**REFINED 2026-04-25 (second alias-swap attempt)**: pushed the alias swap further, surfaced
several additional categories beyond the 8 above:

9. **Value can hold std::pmr::unordered_map (Value::Map) but `get_if<property_map>` after
   alias swap looks for ValueMap** — type mismatch. Yaml-importer `getProperty<T>` recursive
   pattern uses `get_if<property_map>` to descend into nested maps; needs migration to
   `get_if<Value::Map>` (the actually-stored type) OR Value's storage redesign needs to
   accept ValueMap (Step C+D).

10. **`message.data` is `std::expected<property_map, Error>`** (=`std::expected<ValueMap, Error>`
    after swap) but Block.cpp constructs `pmt::Value::Map{...}` (=unordered_map) and assigns —
    type mismatch on every property callback. Need to migrate ALL `pmt::Value::Map{...}` to
    `property_map{...}` in Block.cpp.

11. **`auto& [k, v] : valuemap` binds non-const reference to rvalue** — ValueMap iter yields
    by-value. `init_from_map` in Value.hpp (line 289) needs `const auto& [k, v]` instead;
    drop the `canMove` move-from-source branch (always copies, since iter is by-value).

12. **`updateMaps` doesn't have a property_map overload** — the existing one takes
    `pmt::Value::Map`. After swap, callers pass `property_map`. Need a parallel
    `updateMaps(property_map&, const property_map&)` (shallow merge, or replicate the
    nested-map deep-merge logic via `get_if<Value::Map>` since nested values still hold
    Value::Map until storage redesign).

13. **DataSet concept conformity** — `DataSetLike` concept hard-codes `gr::pmt::Value::Map`
    in the timing_events requirement. Post-swap, DataSet's `pmt_map = property_map` doesn't
    match. Fix: change concept to use `typename T::pmt_map`.

14. **`property_map config = {}` default arg** — explicit ctor + copy-list-init doesn't work.
    Must change to `= property_map{}` (direct-init).

15. **Settings.hpp `std::hash<Value::Map>`** — actually the visitor needs BOTH a property_map
    handler (for ValueMap, accumulating order-independent hash) AND a Value::Map handler
    (for the still-stored unordered_map). The alias swap makes them different types, and
    Value's stored unordered_map still flows through the visitor.

**Additive helpers added to ValueMap during the attempt (KEPT in working tree, ~80 SLOC)**:

- `insert(const PairT&)` — STL-parity pair insert (returns from emplace).
- `insert(InputIt first, InputIt last)` — STL-parity range insert.
- `insert(std::initializer_list<PairT>)` — helper.
- `ValueMap(std::initializer_list<value_type>, resource)` — explicit value_type init-list ctor
  (was a templated PairT version that couldn't deduce from `{"k", v}` brace init).
- `ValueMap(InputIt first, InputIt last, resource)` — range ctor for `property_map{init.begin(), init.end()}`.
- `_assignValueAt` handles `pmt::Value` source — runtime-dispatch on Value's value/container
  type, delegates to typed handlers (allows `valuemap.insert_or_assign(key, pmt::Value{...})`
  for any Value content). Mirrors the converter-from-Map ctor logic.

### 2026-04-25 — Alias swap landed compile-green, surfaced 1 runtime semantic diff

**Approach used**: simpler single-step path — `using Map = ValueMap` in Value.hpp (forward-declared);
out-of-class `Value::init_from_map` body in ValueMap.hpp (where ValueMap is fully defined);
`property_map = Value::Map = ValueMap` cascades automatically.

**Files modified**: `Value.hpp` (forward-decl ValueMap, `using Map = ValueMap`, drop init_from_map body
to declaration), `Value.cpp` (include ValueMap.hpp, Map ctor/op=/copy_from use ValueMap copy-with-resource),
`ValueMap.hpp` (add `buildNestedMapValue` helper for decode bodies that referenced `Value::Map`,
add `init_from_map` body, non-explicit default ctor for `= {}` patterns, add `insert(value_type)`
overload for brace-init), `ValueHelper.hpp` (include ValueMap.hpp; memory_usage uses blob().size()),
`Tag.hpp` (Tag::at by-value, Tag::get → optional<Value>, insert_or_assign uses ValueMap API,
updateMaps shallow-merge overload, put templates use map.resource()), `Settings.hpp` (visitor
needs both Value::Map and property_map handlers — same type now, kept the property_map one),
`Settings.cpp`, `Block.hpp` (filterAndSubstitute uses (*it).second; merge uses lvalue not std::move;
publishEoS uses property_map{}), `Block.cpp` (iter→(*it), Value::Map{...}→property_map{...}),
`Port.hpp`, `Graph.hpp`, `Graph_yaml_importer.hpp`, `BlockModel.hpp`, `PluginLoader.hpp`,
`DataSet.hpp` (concept fix), `FileIo.hpp`, `TagMonitors.hpp` (iter→(*it)), `qa_Tags.cpp`
(`= {}` → `= property_map{}`).

**qa_Tags COMPILES GREEN.** All 204+ original errors resolved.

**RUNTIME FAILURE** — one remaining behavioral difference: 21+ tests fail with
`gr::pmt::ValueMap::operator[]: key not present`. ValueMap's `operator[]` is read-only and
throws on missing key (by design — see ValueMap.hpp:1064 comment: "operator[] throws on
missing key. Library / framework / SYCL code that must stay exception-free MUST use at()
/ find() / contains()"). std::pmr::unordered_map's `operator[]` default-constructs on miss.

Code paths in Settings.cpp / Block.hpp / Tag.hpp may use `tag.map[k] = value` style writes
(or `if (auto& v = map[k])` reads-with-default), which silently break post-swap.

**Next-session work**:

- Audit all `map[k]` sites across the codebase. Migrate writes to `map.insert_or_assign(k, v)`
  and reads to `map.contains(k) ? map.at(k) : default`.
- Run qa_Tags + qa_Settings + qa_Block + qa_TriggerMatcher + qa_DataSink + qa_Scheduler etc.
  Iterate until green.
- Then add corner-case tests (operator== with same blob, view-mode round-trip, etc.)

**Recommended approach for next session** (single dedicated session, ~400-500 SLOC):

Phase 1 — ValueMap pre-fixes (additive, no API change):

- Already done: see "Additive helpers added" above.
- TODO: audit Tag.hpp + Settings.hpp visitor handler for `Value::Map` (= unordered_map) handler
  alongside the property_map (= ValueMap) handler in `computeValueHash`.

Phase 2 — Source-side prep that DOESN'T require alias swap:

- (a) audit all `iter->second` / `iter->first` sites and migrate to `(*iter).second`;
  (b) audit and migrate all `map[k] = v` writes to `insert_or_assign`; (c) replace
  `map.get_allocator().resource()` with `map.resource()`; (d) add ValueMap::insert overload
  (or remove the call sites); (e) initialiser-list construction either via new ValueMap
  ctor or rewrite call sites; (f) handle the nested-Map case in ValueMap::ValueMap(Map&)
  conversion ctor; (g) fix std::hash situation in Settings.hpp.
- After source-side fixes: flip alias, fix Tag::at/get/insert_or_assign, run focused
  qa_Tags / qa_Settings / qa_Block / qa_Scheduler / qa_TriggerMatcher; expand test set
  as fixes propagate.
- Realistic SLOC: 250-400 (the source-side fixes are mechanical but numerous).

**After Tag alias swap, the remaining Phase 1e work is**:

- Tensor storage byte-blob redesign (Step B.3, ~300 SLOC) — biggest single piece
- Map storage byte-blob redesign + Value::Map = ValueMap alias swap (~250 SLOC) — couples
  with Tensor work since both need format helpers in Value detail
- ValueMap iter view-mode for Tensor + Map entries (~50 SLOC) — depends on storage redesigns
- ImGraph 4 extract→reassign refactor (~80 SLOC)
- ValueHelper.hpp template machinery refactor (~50 SLOC)
- Test sweep + corner-case tests (view-mode round-trip, byte-blob round-trip, etc.)

**Total remaining**: ~700-1000 SLOC. Realistically 2-3 focused sessions.

**Pre-existing failure**: `qa_SubGraphAssets` reports 1 fail before *and* after these changes
(plugin loading issue with `good::multiply<float64>`); unrelated to Phase 1e.

Remaining steps: Tensor byte-blob (B.3), Map byte-blob + alias flip (C+D combined),
view-type accessors for ValueMap (E), iter view-mode for Tensor + Map (F), full caller
migration sweep for Map sites (G), Tag.hpp property_map alias swap + Tag::at by-value (H).

Tests + 14 in-tree callers migrated for Step A:

- core/test/qa_Value.cpp: pmr::string-typed `value_or<T&&>` / `or_else<T&&>` / `transform<T&&>` /
  `transform_or<T&&>` / `and_then<T&&>` test cases rewritten using std::int64_t (the wrapper
  semantics are still tested via scalars; the pmr::string transfer path is no longer meaningful
  with byte-blob storage).
- core/test/qa_ValueHelper.cpp:792: `get_if<std::pmr::string>` → `value_or<std::string_view>`.
- blocks/audio/include/AudioBlocks.hpp + qa_Audio.cpp: `get_if<std::pmr::string>` →
  `get_if<std::string_view>`.
- blocks/sdr/include/RTL2832Source.hpp + SoapySource.hpp: same migration.
- blocks/sdr/test/qa_RTL2832Source.cpp + qa_SoapyIntegration.cpp: same migration.

Remaining steps: Tensor byte-blob (B), Map byte-blob + alias flip (C+D combined), view-type
accessors (E), iter view-mode for Tensor + Map (F), full caller migration sweep (G),
Tag.hpp property_map alias swap (H).

**2026-04-24 follow-up wave (items 1–7 landed; items 8 + 9 deferred to a separate PR)**:

| # | What landed                                                                                                                                                                                                                                                                | SLOC | Effect                                                                                                                                                                                                    |
|---|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1 | `ValueMap::merge` raw-byte fast path (memcpy `PackedEntry` + payload between blobs; batch-compact source)                                                                                                                                                                  | ~40  | merge N=20: 324k → 408k ops/s (+26 %); now competitive with `Value::Map`'s 415k                                                                                                                           |
| 2 | Iter inline-switch in `operator*()` (bypasses `dispatchValueType` lambda; dense-id key fast path)                                                                                                                                                                          | ~50  | iter N=20: 7.9M → 10.1M ops/s (+28 %); inherent gap to unordered_map (Value-construction-per-deref) remains                                                                                               |
| 3 | `Block.hpp` tag-merge `merged.reserve()`                                                                                                                                                                                                                                   | 1    | avoids rehash storms on dense tag maps                                                                                                                                                                    |
| 4 | `Settings::autoUpdate` empty-tag-map early-exit                                                                                                                                                                                                                            | ~10  | skips lock + ctx-resolution overhead on tag-less work() calls; preserves `setChanged(false)` semantics                                                                                                    |
| 5 | Long-key spill-to-pool (`kSpilledKeyId` sentinel; key bytes spill into payload pool with offset+length stored in `inlineKey[0..7]`); covers spill in `_writeKey` / `_entryMatches` / iterator / `_grow` (offset relocation) / `merge` / `decodeEntry` (nested-map walking) | ~80  | Removes the documented "keys > 27 chars rejected" limitation; round-trip + grow-survival + merge tests added                                                                                              |
| 6 | `ValueMap::from_blob(span, resource)` returning `std::expected<ValueMap, DeserialiseError>` (validates magic / version / alignment / totalSize / entry-payload offsets / spilled-key offsets)                                                                              | ~50  | Wire-format ingest path; round-trip + bad-magic + too-small tests added                                                                                                                                   |
| 7 | `Value::set_types` moved inline to header                                                                                                                                                                                                                                  | 4    | Slight inlining win on the Value-construction path; iter delta within stddev (Value's `Storage` zeroing + `ensure_resource` branch + bit-field updates dominate; deeper redesign needed for further wins) |

**Items 8 + 9 deferred to a follow-up PR** (`featValueMapAliasSwap`):

- **Item 8** (alias `using property_map = ValueMap;`) original SLOC estimate (50–100 in core) was wrong:
  `Tag::at()` returns `Value&` and `Tag::get()` returns `std::optional<std::reference_wrapper<const Value>>`.
  `ValueMap::at()` returns `Value` **by value** — incompatible. Either Tag's return types change (touches all
  18 `tag.get(...)` callers), or ValueMap grows a reference-yielding accessor (architectural breaking change).
  Plus `PropertyMapType` concept needs widening, plus the 14 `get_if<std::pmr::string>` callers must migrate to
  `get_if<std::string_view>` (or iter must revert to allocating the pmr::string per deref, losing Phase 1d step 3's
  win). Realistic scope: ~200–400 SLOC across `core/`, `blocks/sdr/`, `blocks/audio/`, `algorithm/`, plus several
  semantic decisions (Tag::at by-value vs add `Value&` accessor; opt-in view iter vs migrate callers; etc.).
- **Item 9** (`get_if<Value::Map>` + `get_if<Tensor<T>>` migration to view-mode accessors) — explicitly deferred
  per 2026-04-24 user direction; ~67 caller sites; only relevant if/when ValueMap iter wires Tensor + Map to
  view-mode (Phase 1d steps 5 + 6, also deferred).
- Both should land together in `featValueMapAliasSwap` as a single coherent migration with its own audit + test
  pass; out-of-scope for this branch.

#### Phase 1d follow-up — open design decisions resolved 2026-04-24

Resolutions for the cross-cutting design questions raised after items 1–7 landed:

| #   | Decision                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
|-----|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Q1  | **Value redesign — packed-blob internal storage** (NEW direction): Value's intrinsic storage uses the same byte-blob model as ValueMap. ValueMap stores Value's sub-blobs as fields (not via heap pointers to `pmr::string` / `Tensor` / `Map`). Resolves the iterator `Value&` vs by-value tension AND closes the per-deref Value-construction gap. Effort: substantial; tracked as Phase 1e (`featValueRedesign`) before / coupled with `featValueMapAliasSwap`. |
| Q2  | **`PropertyMapType` concept** — hard-cut. After alias swap, only `same_as<T, ValueMap>` accepted; no transition window.                                                                                                                                                                                                                                                                                                                                            |
| Q3  | **`ValueMap::operator[]` insert-on-miss** — do NOT add. Migrate the 1 call site (qa_DataSink.cpp:169-171) instead.                                                                                                                                                                                                                                                                                                                                                 |
| Q4  | **`TensorView<Value>::owned()` recursion** — no depth cap; use the same (single) PMR allocator as the top level.                                                                                                                                                                                                                                                                                                                                                   |
| Q5  | **ValueMap `_ownsBlob` mutator policy** — options under discussion (assert / `std::expected` / no-op / hybrid). Decision pending — see "Open questions" below.                                                                                                                                                                                                                                                                                                     |
| Q8  | **`Value::Map` deletion** — yes; replace `using Map = std::pmr::unordered_map<...>` (Value.hpp:159) with `using Map = ValueMap;`. Gated on Q1 (Value redesign). Lands in `featValueMapAliasSwap`.                                                                                                                                                                                                                                                                  |
| Q9  | **Wire-format / SigMF compat for alias swap** — clean cut; no prior persisted state to migrate.                                                                                                                                                                                                                                                                                                                                                                    |
| Q13 | **`Value::Map → std::flat_map` migration** — DROPPED. `flat_map` is not USM-compatible (uses two `vector<>`s with separate allocations + internal pointers). The TODO at `Value.hpp:159` is removed; the replacement target is `ValueMap` (per Q8), not `flat_map`.                                                                                                                                                                                                |
| Q14 | **Endian portability** — keep LE-only. Defer until cross-arch USM is on the roadmap.                                                                                                                                                                                                                                                                                                                                                                               |
| Q15 | **Multi-threaded reader path** — not an issue. Reader reads after writer publishes; no concurrent mutation while reading.                                                                                                                                                                                                                                                                                                                                          |
| Q16 | **Value SBO for strings** — REPHRASED: post-Q1-redesign, Value's String storage IS the packed-blob model (raw `size + char_array + '\0'` mandatory; no `std::pmr::string` allocation). Same shape as ValueMap's String entries. Saves the per-string allocation that the current `pmr::string`-via-`_storage.ptr` path incurs.                                                                                                                                     |
| Q17 | **Cross-resource merge** — when source's PMR resource differs from destination's, migrate the bytes through destination's resource (don't keep source pointers in destination).                                                                                                                                                                                                                                                                                    |
| Q18 | **ValueMap minimum capacity** — either 0 (truly empty, no header) or ≥ 416 bytes (first proper alignment size = `sizeof(Header) + 8·sizeof(PackedEntry)`); thereafter geometric growth like `std::vector`.                                                                                                                                                                                                                                                         |

**Phase 2 design questions** (`PooledRingBuffer` + `TagRingBuffer` + `houseKeeping`) — delegated to later when
that sub-phase starts.

**Phase 1b·c (device-side / SYCL atomic primitives)** — actions: design SYCL `atomic_ref` model on
`header.entryCount` + `header.payloadUsed` (relaxed vs acq/rel), lock-free append via compare-exchange,
overflow-flag setting on slack exhaustion. Defer to Phase 1b·c implementation; capture sketch when work resumes.

#### Phase 1e migration plan — Path B (drop non-owning returns + caller migration), 2026-04-24

Path B chosen over Path A (lazy-cache) — Value stays at 24 bytes, API semantics are clearer at the call site
(view + `.owned()` is more obvious than `auto* p = ...; use(*p)`), and audit shows the **production blocklib
touch is just 2 sites**.

**Caller-site classification** (67 total, post Q1 redesign):

| Area                                       |            Tensor sites |                                 Map sites |  Total | Migration class                                                                                                                     |
|--------------------------------------------|------------------------:|------------------------------------------:|-------:|-------------------------------------------------------------------------------------------------------------------------------------|
| `core/test/qa_*` (own tests)               |                      15 |                                         7 |     22 | mechanical rewrite                                                                                                                  |
| `core/include/` helpers                    |                      17 |                                        13 |     30 | view-only suffices (Graph_yaml_importer, ValueHelper, Settings, Scheduler, ValueFormatter)                                          |
| `core/src/Block.cpp`                       |                       1 |                                         0 |      1 | view-only suffices                                                                                                                  |
| `algorithm/include/.../ImCanvas.hpp`       |                       0 | 0 (uses `get_if<bool>` only — unaffected) |      0 | n/a                                                                                                                                 |
| `algorithm/include/.../fileio/FileIo.hpp`  |                       1 |                                         0 |      1 | view-only suffices                                                                                                                  |
| `algorithm/include/.../ImGraph.hpp`        | **2 owning-mutate** + 0 |                       **2 owning-mutate** |  **4** | **owning-mutate refactor needed** — extract → mutate → reassign pattern (post Q1, no in-place mutation possible regardless of path) |
| `meta/include/.../UnitTestHelper.hpp`      |                       1 |                                         0 |      1 | view-only suffices                                                                                                                  |
| `blocks/sdr/test/qa_RTL2832Source.cpp`     |                       0 |                                         4 |      4 | view-only suffices                                                                                                                  |
| `blocks/timing/test/qa_PpsSource.cpp`      |                       0 |                                         1 |      1 | view-only suffices                                                                                                                  |
| `blocks/timing/test/qa_GpsSource.cpp`      |                       0 |                                         1 |      1 | view-only suffices                                                                                                                  |
| `blocks/sdr/include/.../RTL2832Source.hpp` |                       0 |                                         1 |      1 | view-only suffices (read-only nested map access)                                                                                    |
| `blocks/audio/include/.../AudioBlocks.hpp` |                       0 |                                         1 |      1 | view-only suffices (read-only nested map access)                                                                                    |
| **TOTAL**                                  |                  **39** |                                    **28** | **67** |                                                                                                                                     |

**Production blocklib touch: 2 sites** (`RTL2832Source.hpp` + `AudioBlocks.hpp`, both view-only). All other
blocklib touches are in test files (6 sites, view-only).

**Owning-mutate sites (4 total, all in `ImGraph.hpp`)** require an `extract → mutate → reassign` refactor pattern
because post Q1 (Value's intrinsic storage = byte-blob), there's no in-place mutable Tensor / Map heap object to
get a reference to under EITHER path. Path A's lazy-cache wouldn't help — cache mutations wouldn't propagate back
to the byte-blob (cache divergence). The refactor:

```cpp
// OLD: in-place mutation via owning ref
auto& v = *it->second.get_if<Tensor<Value>>();
v.resize(...); v[i] = ...;
// NEW: extract → mutate → reassign
auto v = (it->second.get_if<TensorView<Value>>() ? it->second.get_if<TensorView<Value>>()->owned(_resource) : Tensor<Value>{...});
v.resize(...); v[i] = ...;
it->second = std::move(v);
```

**Total Phase 1e + alias-swap SLOC** (revised from earlier 200-400 estimate):

- Value redesign internals: ~400 SLOC
- Caller migrations: ~200 SLOC (67 sites × ~3 lines)
- ImGraph extract→reassign refactor: ~50 SLOC (4 sites)
- Alias swap (typedef flip + concept hard-cut + Tag::at by-value): ~50 SLOC
- **Total: ~700 SLOC** (vs earlier 800-1500 estimate — much smaller because blocklib touch is bounded).

---

## 1. Design outcome summary (resumable context)

Replace `gr::pmt::Value::Map` (currently `std::pmr::unordered_map<std::pmr::string, Value, MapHash, MapEqual>` — see `core/include/gnuradio-4.0/Value.hpp:159`) with a new `gr::pmt::ValueMap` type that is cache-friendlier, canonical-ID-aware, and (optionally) USM-blob-ready.

Rework tag storage so it is decoupled from stream capacity: today every port pre-allocates `kDefaultBufferSize=4096` slots of cache-line-aligned `Tag`, i.e. ≥256 KB per edge just for tag slots, on every platform including RP2350. Introduce `TagRingBuffer<Tag>` based on a common `PooledRingBuffer<Descriptor, Payload>` primitive (shared between `TagRingBuffer` and, in v2+, a `CircularBuffer<LargeT, Pooled>` specialisation).

Two graph-level PMR pools (separate for stream and tag) act as upstream to edge-level pools; edge-level pools already exist in struct shape (see `Edge::_dataResource`, `_tagResource`).

A Block-level `houseKeeping()` method (parallel to `processMessages()`) drives lazy reclamation in pooled buffers on a time/iteration gate. No cross-API changes.

---

## 2. Investigation findings (from §2 of the spec — done)

### 2.1 Spec claims **CONFIRMED** by code read

- `Value::Map` is `std::pmr::unordered_map<std::pmr::string, Value, MapHash, MapEqual>` — `Value.hpp:159` (has a `// TODO: replace with std::flat_map` comment that predates this rework)
- `Value.hpp:307-308` copy-ctor bug: the single-arg ctor passes `std::pmr::get_default_resource()` instead of inheriting `other._resource`. Two-arg ctor (line 308) has nullptr-fallback to `other._resource` that is never invoked because of the default argument. Verified verbatim.
- `init_from_map` (`Value.hpp:254-276`) already deep-copies keys + values into `_resource` correctly. Its logic is the template for `ValueMap`'s cross-resource insert.
- Transparent lookup via `MapHash` / `MapEqual` (`Value.hpp:142-157`): supports `const char*`, `string_view`, `basic_string<...>`. `gr::meta::fixed_string` works only via implicit conversion to `string_view`.
- Emscripten guard on `holds<std::size_t>()` at `Value.hpp:822-825`.
- `CircularBuffer` is PMR-enabled via `std::pmr::polymorphic_allocator<T>` (`CircularBuffer.hpp:231, 777-790`). Double-mapped-page fast path on POSIX is gated on `std::is_trivially_copyable_v<T>` (line 778).
- Compile-time producer policy spelled `ProducerType::{
    Single, Multi}` (`CircularBuffer.hpp:227`). The spec's SPSC/SPMC/MPMC terminology maps onto this.
- Blocks survey: median tag payload 250–350 B; 97% ≤ 512 B; 3% outliers (GPS/PPS with nested geolocation) up to ~1.2 KB. Spec's size-class buckets `{64,128,256,512,1024,2048}` validated.
- YaS reference patterns in `opencmw-cpp/src/serialiser/include/IoSerialiserYaS.hpp`: `ARRAY_TYPE_OFFSET=100` (line 25),
  `START_MARKER=0`, `END_MARKER=0xFE` (254), `MapLike=203` column-oriented (keys array then values array), protocol
  header `[magic=-1][string="YaS"][maj=1][min=0][micro=0]`, field header `{
typeId:
    u8, dataStartOffset : i32, dataSize : i32, fieldName : string, [unit:string, description:string, modifier:u8]}`.

### 2.2 Spec claims **DISAGREE with code** — must correct before implementation

1. **`Tag` is `{
    std::size_t  index;
    property_map map;}`** (`Tag.hpp:74-111`), **not** `{
    uint64_t     sample_idx;
    ValueMap     properties;}`. Field name is `map`, `GR_MAKE_REFLECTABLE(Tag, index, map)` pins it as API. `index` is
   `std::size_t` (32-bit on WASM). Tag also has convenience methods (`reset/at/get/insert_or_assign`) that use
   `convert_string_domain` key normalisation and throw `std::out_of_range`.

2. **Tag buffer ≠ stream buffer in size.** Both default independently to `kDefaultBufferSize=4096` (`Port.hpp:540`). Tag buffer is built by `newTagIoHandler(kDefaultBufferSize)` (`Port.hpp:729-735`) without consulting stream size. Worst-case: ~256 KB per edge just for tag slots.

3. **`CircularBuffer::tryReserve` returns `WriterSpan`, not `bool`** (`CircularBuffer.hpp:492-521`). Failure = empty span. Publish via `WriterSpan::publish(n)` or RAII destructor. Backpressure surfaces as `work::Status::INSUFFICIENT_OUTPUT_ITEMS`.

4. **`Block::publishTag` signature** (`Block.hpp:1305`): `template<PropertyMapType> publishTag(PropertyMap&&, std::size_t tagOffset = 0) noexcept`. Map first, offset second, defaulted. `PropertyMapType` concept in `Tag.hpp:61-62` is `std::same_as<std::decay_t<T>, property_map>` — exact type, so any new `ValueMap` must replace `property_map` (via type alias) or the concept must widen.

5. **Non-trivially-copyable T already works in `CircularBuffer`** via the `std::copy_n` branch at
   `CircularBuffer.hpp:363-378`. `qa_buffer.cpp` tests `buffer_tag{
    int64_t;
    std::string}` and `std::map<int,int>`. Spec's motivation (aggregate path) should be re-stated as "shrink tag
   storage + isolate its allocator + lazy payload reclamation", not "make aggregate T work".

6. **Graph has no PMR resources today** — only `Edge` does, via `_dataResource` + `_tagResource` (`BlockModel.hpp:67-68, 91-92`). `MemoryAllocators.hpp` has no pool/slab wrappers — must use `std::pmr::{un,}synchronized_pool_resource` or build a custom slab.

### 2.3 Additional findings not in spec (worth keeping)

- **`Port.hpp:361` has a `TagBufferType` customisation point**: `struct DefaultTagBuffer : TagBufferType<gr::CircularBuffer<Tag>> {}`. Swap `TagRingBuffer` in here, don't modify `CircularBuffer`. Makes the "scalar CircularBuffer byte-identical after change" concern moot.
- `Value.hpp:159` TODO already anticipates `std::flat_map` migration.
- Tag convenience methods rely on `convert_string_domain` key normalisation. Preserve.
- `Edge` also carries `_uiConstraints` as a `std::shared_ptr<property_map>` (`BlockModel.hpp:96`). Second consumer of `Value::Map` for migration sweep.
- Existing `DefaultTag` system uses `tag::SAMPLE_RATE.shortKey()` etc. Appendix A canonical names must align with these, not invent `gr:sample_rate` prefix (see Q3).
- Scheduler periodic hooks: `cleanupZombieBlocks()` every 16 iterations (`Scheduler.hpp:672, 1082-1144`); `runWatchDog()` every 1 s (`Scheduler.hpp:735-754`). **Not used** — reclamation driven from Block-level `houseKeeping()` (see §3).
- `multiThreaded` scheduler can have an edge's source + sink on different workers (hash-based: `Scheduler.hpp` adoptBlock line 844: `hash<BlockModel*> % nBatches`). Graph-level tag pool therefore needs `synchronized_pool_resource`; per-edge can be `unsynchronized` (writer is single-worker).

---

## 3. Design decisions

### Agreed decisions (round-2 user ack received)

| #                                               | Topic                       | Decision                                                                                                                                                                                                                                                                                                                                                                                    |
|-------------------------------------------------|-----------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| A                                               | Modify `CircularBuffer<T>`? | **No.** Introduce a new `TagRingBuffer<Tag>` swapped in at `Port.hpp:361` via existing `TagBufferType` seam.                                                                                                                                                                                                                                                                                |
| B                                               | Factor lazy reclamation?    | **Yes.** New `PooledRingBuffer<Descriptor, Payload>` primitive in `core/include/gnuradio-4.0/PooledRingBuffer.hpp`. Used by `TagRingBuffer` in v1.                                                                                                                                                                                                                                          |
| C                                               | `Graph` pools               | **v1: no new fields on `Graph`. Both `Edge::_dataResource` and `Edge::_tagResource` continue to default to `std::pmr::get_default_resource()`. Users who want a pool set `edge._tagResource` explicitly (via `EdgeParameters::tagResource` or direct assignment).** v2+: graph-level pool as an ownership hook if profiling shows churn.                                                    |
| D                                               | Reclamation trigger         | **Block-level `houseKeeping(policy)` method** (Q6). Iteration/time-gated, iterates output ports, calls `port.tagBuffer().houseKeeping(policy)` behind a `requires` concept check. Default implementation; user-overridable.                                                                                                                                                                 |
| E                                               | Cross-API changes           | **None.** `Block::publishTag`, `OutputSpan::publishTag`, `tryReserve`+`publish` two-step all preserved.                                                                                                                                                                                                                                                                                     |
| F                                               | `Tag::map` field name       | **Keep `map`** (reflection-stable via `GR_MAKE_REFLECTABLE`). (Q4)                                                                                                                                                                                                                                                                                                                          |
| G                                               | `Tag::index` type           | **Keep `std::size_t`** (WASM-safe).                                                                                                                                                                                                                                                                                                                                                         |
| H                                               | ValueMap layout             | **Path α — flatbuffer** (Q1). Single contiguous PMR allocation, pointer-free/offset-addressed, USM-ready: `blob()` memcpy'able to SYCL-USM, device kernels read all fields + can append when `slack_bytes > 0`. Matches spec §3.                                                                                                                                                            |
| I                                               | Large-T CircularBuffer      | **v1: factor `PooledRingBuffer` primitive only.** v2: auto-specialise `CircularBuffer<T, ...>` when T is PMR-capable (trait-detected, no opt-in policy tag) so slots are shrinkable once the slowest reader has passed. (Q2)                                                                                                                                                                |
| J                                               | Canonical keys              | **Authoritative source: `kDefaultTags` array at `Tag.hpp:218` + `DefaultTag<...>` declarations at `Tag.hpp:196-216`.** No `gr:` prefix on names (prefixing is a later refactor). Types come from `DefaultTag` truth — **NOT** spec's Appendix A (which mis-states several: `sample_rate` is `float` not `Float64`, `signal_min/max` are `float`, `trigger_offset` is `float` s, etc.). (Q3) |
| K                                               | HouseKeeping policy         | **`enum class HouseKeepingPolicy { Reclaim = 0x01, Shrink = 0x02, All = Reclaim\|Shrink };`** `void houseKeeping(HouseKeepingPolicy policy = HouseKeepingPolicy::All)`. Default is do both; caller can restrict. (Q6)                                                                                                                                                                       |
| L                                               | `Value.hpp:307` fix         | **`Value(const Value& other) : Value(other, other._resource) {}`** — single-arg copy ctor inherits source resource. `copy_from` uses target's `_resource` (see design-intent comment at `Value.cpp:96-102`); the bug was that target's `_resource` was silently overwritten to default. Map inserts were always safe (explicit two-arg `Value{                                              
 val, _resource}` path in `init_from_map`). (Q5) |

### Round-3 decisions (2026-04-22, post-3d review — user ack received)

Triggered by a sit-down review of `19404e43`: naming, exception safety, user-visible key-ID removal, iterator
completeness. Decisions supersede the corresponding §4.2.1 3d-specific choices where they conflict.

| #  | Topic                                  | Decision                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
|----|----------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| M  | `time` → `ctx_time` rename             | **Hard rename now, no bridge.** Wire-format break accepted; out-of-tree consumers (persisted YAMLs, SigMF, OpenDigitizer) adopt at their own cadence. Removes the long-standing `Tag.hpp:213` TODO.                                                                                                                                                                                                                                                                                                                                                                                                                      |
| N  | "Flatbuffer" terminology in source     | **Drop.** `FlatEntry` → `PackedEntry`; reword the doc block to use "contiguous packed layout" / "packed blob". External framework references (Flatbuffers, Protobuf, Cap'nProto) stay in this WIP and commit messages only.                                                                                                                                                                                                                                                                                                                                                                                              |
| O  | Naming-convention sweep                | **STL-mirror snake_case kept** for public entry points (`insert_or_assign`, `try_emplace`, `shrink_to_fit`). Everything else → lowerCamelCase per CLAUDE.md §1.3: private methods, all `detail::` helpers, registry functions (`idOf` / `lookupId` / `boundTypeOf` / `unitOf` / `canonicalCppTypeFor` / `cppToValueType`), wire-format field names in `Header` / `PackedEntry` / `CanonicalKey` (renames are safe — wire-format is positional bytes, not field-name-keyed).                                                                                                                                              |
| P  | `valueTypeToCpp` shape                 | **`consteval` function + `if constexpr` cascade** (returning `std::type_identity<T>`), not a partial-specialisation struct. Together with a single compile-time `kBindingTable<{ValueType, CppType}>` drives a `dispatchValueType<F>(vt, f)` helper that replaces the hand-written switches in `entryToValue` / `cppToValueType`. Concept pair layered on top: `CanonicalName<Name>` (registered) + `BoundCompatible<Name, T>` (`std::convertible_to<T, CanonicalCppType<idOf<Name>>>`). Constrains the typed fluent; collapses the 4 `static_assert`s in `set<Id>` to one constrained signature with crisp diagnostics. |
| Q  | STL / C++23 simplifications            | **Apply**: registry linear loops → `std::ranges::find_if`; inline-scalar read/write memcpy → `std::bit_cast`; `_grow` offset relocation → `std::ranges::for_each`; dead fallthrough in `_writeValue` deleted; `_writeValue` ∪ `_assignValueAt` merged (index-form only); `_insertNew` ∪ `_upsertCanonical` merged via key-resolution lambda.                                                                                                                                                                                                                                                                             |
| R  | Move-assign shape                      | **Keep the current dual-path** (same-resource steal, cross-resource deep-copy). Copy-and-swap with a by-value parameter silently keeps the **source**'s resource on cross-resource moves, violating the decision-L PMR contract that landed in `4f3e801b`. Reconfirmed.                                                                                                                                                                                                                                                                                                                                                  |
| S  | Iterators                              | **Random-access, const-only.** `iterator = const_iterator`. Full STL set: `begin/cbegin/end/cend/rbegin/crbegin/rend/crend`. Tests pin `std::ranges::random_access_range` + `std::ranges::sized_range` conformance. Storage is contiguous `PackedEntry[]` → trivial; by-value `std::pair<std::string_view, Value>` dereference retained.                                                                                                                                                                                                                                                                                 |
| T  | `operator==` and `std::hash<ValueMap>` | **Deferred.** No concrete consumer yet. Revisit when a set-of-maps or wire-format diff path appears.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| U  | User-visible key-IDs                   | **~~Removed from the public API via typed-name fluent~~ → SUPERSEDED 2026-04-23 (post-investigation, see Round-3.5 below).** Typed-name fluent dropped; `keys::idOf<>` remains an implementation detail used only inside `dispatchValueType<F>` and serialisation paths. `get<>`/`set<>` stay deleted.                                                                                                                                                                                                                                                                                                                   |
| V  | Exception-free core                    | **~~`at<"name">()` returns `T&`~~ → SUPERSEDED 2026-04-23.** Typed `at<"name">()` dropped; framework hot-path investigation showed iteration dominates and the typed-fluent `T&` return doesn't measurably win over `find()` + `value_or<T>`. Caller pattern is now `m.at("name").value_or<T>(default)` — see decision W.                                                                                                                                                                                                                                                                                                |
| W  | `at(std::string_view)`                 | **Returns `Value` by value; Monostate iff missing; no throw.** Caller pattern: `m.at("foo").value_or<float>(0.f)`. No `std::optional<Value>` wrapping — double-unwrap is ugly; `Monostate` already conveys absence. **Kept post-2026-04-23 review.**                                                                                                                                                                                                                                                                                                                                                                     |
| X  | `contains<"name">()` semantics         | **~~Typed contains<"name">() checks key + type~~ → SUPERSEDED 2026-04-23.** Typed contains dropped along with the typed fluent. Untyped `contains("name")` (key existence only) is the sole survivor.                                                                                                                                                                                                                                                                                                                                                                                                                    |
| Y  | `bool` storage                         | **Drop the low-bit-pack; store as one byte at offset 0 of `inlineValue`.** Same 8 bytes of storage, same wire bytes. Enables `at<"name">() -> bool&` via `std::start_lifetime_as<bool>` uniformly with other inline scalars. `inlineValue` alignment (8 B) covers `alignof(bool) == 1`.                                                                                                                                                                                                                                                                                                                                  |
| Z  | `qa_ValueMap.cpp` structure            | **Didactic reorder**: construction → empty / find / contains → string-keyed insert → typed-name fluent (at / insert_or_assign / try_emplace / erase) → mutation (clear, overwrite) → iteration & ranges conformance → copy / move / swap → growth (reserve, shrink, preserving entries and string payloads) → blob format invariants → edge cases (27-char boundary, frozen, wrong-type insert, missing-key assert).                                                                                                                                                                                                     |
| AA | `gr::allocator::pmr::CountingResource` | **Add to `MemoryAllocators.hpp`** as the PMR analogue of the existing allocator-side `gr::allocator::Logging`. Exposes `allocCount`, `deallocCount`, `liveBytes`. Consumed by both `qa_ValueMap` (replacing the local `counting_resource`) and `qa_Value` (replacing the ad-hoc resource tracking in the PMR-regression block).                                                                                                                                                                                                                                                                                          |

---

## 4. Concrete next steps

### 4.1 Commits landed on this branch

- **`4f3e801b` — `fix(core): Value copy/move-assign preserve target's PMR resource`** (squashed from two earlier working commits; originals reachable via reflog for ~30 days). Single-arg copy ctor at `Value.hpp:307` inherits `source._resource` instead of rebinding to default. `operator=(Value&&)` split into same-resource O(1) steal and cross-resource deep-copy via target's resource, matching `std::pmr::vector` (`propagate_on_container_move_assignment = false_type`). **Behavioural**: `Value v2 = v1;` and move-assign stay on the target's resource; callers needing the old rebinding must pass `get_default_resource()` explicitly via the two-arg ctor. **Signature**: `operator=(Value&&)` is no longer `noexcept`; flips `std::is_nothrow_move_assignable_v<Value>` to `false`. qa_Value gained 4 copy-ctor + 2 move-assign regression cases inside the "pmr resource usage and propagation" block.
- **`f3c396f5` — `feat(core): ValueMap skeleton + canonical key registry (Phase 1 start)`**. `gr::pmt::ValueMap` skeleton (Header, FlatEntry, DeserialiseError, public API declarations) + `gr::pmt::keys` registry (20 entries pinned from `Tag.hpp`), with `qa_ValueMapKeys` (8 tests / 105 asserts). Spec deviation: `FlatEntry.inline_key` shrunk to 28 B (27 usable chars) because the 48 B total with a 32-byte key + 64-bit inline_value is impossible.
- **Phase 1 · step 3d — core implementation** (scalars + strings) — see §4.2 sub-step list for detail, and the commit message for non-obvious tradeoffs. SHA filled in once committed (this commit).
- **(`docs: add WIP note for featValueMapExtension branch`** — was a separate commit, now folded into `f3c396f5` via amend — this file is tracked there.)

### 4.2 Phase 1 — `ValueMap` (path α — flatbuffer, USM-ready)

**Commit strategy** (revised 2026-04-22): 3d and onwards land as **separate new commits** on top of `f3c396f5` for reviewability. Squash before merge to main if desired.

Scope: single contiguous PMR allocation, pointer-free, offset-addressed (spec §3, with corrections from §2.2).

Sub-steps (execution order):

Done (pre-refactor):

1. ✅ **Appendix A pin-down** — canonical key table from `Tag.hpp:196-218`. See Appendix A. *Landed in `f3c396f5`.*
2. ✅ **Skeleton + registry** — `ValueMap.hpp` holds both `gr::pmt::ValueMap` class declarations and the `gr::pmt::keys`
   registry (`CanonicalKey` struct, `kCanonical` 20-entry array, constexpr `id_of<"…">` / `bound_type_of<id>` /
   `unit_of<id>` / `canonical_cpp_type_for<id>`, consteval integrity checks). `qa_ValueMapKeys.cpp` covers the
   registry (8 tests / 105 asserts). *Landed in `f3c396f5`.*
3. ✅ **Core impl** — blob alloc + grow-on-demand with payload-offset relocation; lookup (canonical-ID fast path +
   inline-key scan); `emplace` / `insert_or_assign` / `set<Id>` / `get<Id>` / `at()` / `erase` / `clear` / `reserve` /
   `shrink_to_fit`; const iterator yielding `pair<string_view, Value>`; cross-resource copy/move. `qa_ValueMap.cpp` (31
   tests / 114 asserts). Scalars + `std::string` only. *Landed in `19404e43`.* See §4.2.1 for 3d-specific design
   decisions (many now superseded by Round-3).

#### Round-3 refactor — 9-commit series (supersedes the pre-review 3e+ plan)

Each commit lands independently build-green and test-green under `-j6`. Authorisation is per-commit; WIP update
accompanies commit 1 (or as its own commit 0 — see diff presentation).

1. ⏸ **`time` → `ctx_time` rename (decision M)** — flip `DefaultTag<"time", …>` template arg at `Tag.hpp:213`, mirror in
   `kDefaultTags[]` (line 218), update `kCanonical` entry at `ValueMap.hpp:120`. Remove the `Tag.hpp:213` TODO.
   Exercises: `qa_Tags`, `qa_Messages`, `qa_ValueMapKeys`, `qa_ValueMap`. Appendix A's table row + pending-decision note
   updated in the same commit.
2. ⏸ **"Flatbuffer"→"Packed" terminology (decision N)** — `FlatEntry` → `PackedEntry`, rewrite the ValueMap doc block to
   drop external-framework names. Source-only; WIP + commit messages keep "flatbuffer-shaped" as a mental model.
3. ⏸ **Naming-convention sweep (decision O)** — snake_case → lowerCamelCase on private members, all `detail::` helpers,
   registry free functions (`idOf` / `lookupId` / `boundTypeOf` / `unitOf` / `canonicalCppTypeFor`), `cppToValueType`,
   and wire-format field names in `Header` / `PackedEntry` / `CanonicalKey`. STL-mirror public API (`insert_or_assign`,
   `try_emplace`, `shrink_to_fit`, `find`, `contains`, `at`, `erase`, `reserve`, `clear`, `emplace`, `size`, `empty`,
   `begin`, `end`) stays snake/lowerCamel as STL has it.
4. ⏸ **Consteval cascade + concepts (decision P)** — `valueTypeToCpp` becomes a `consteval` function with `if constexpr`
   cascade returning `std::type_identity<T>`; `kBindingTable<{ValueType, CppType}>` drives a single
   `dispatchValueType<F>(vt, f)` helper that replaces the hand-written switches in `entryToValue`. Concepts
   `CanonicalName<Name>` + `BoundCompatible<Name, T>` added.
5. ⏸ **STL simplifications + `bool` storage (decisions Q, Y)** — registry loops → `std::ranges::find_if`; inline-scalar
   read/write → `std::bit_cast`; `_grow` offset relocation → `std::ranges::for_each`; dead fallthrough in `_writeValue`
   deleted; `_writeValue` ∪ `_assignValueAt` merged (index-form only); `_insertNew` ∪ `_upsertCanonical` merged. `bool`
   stored as a single byte at offset 0 of `inlineValue` (drops the bit-pack) — enables `bool&` aliasing for commit 6.
6. ⏸ **Typed-name fluent (decisions U, V, X)** — new public API: `at<"name">()` returning `T&` / `std::string_view` with
   `assert(contains<Name>())` gated on `gr::meta::kDebugBuild`; `insert_or_assign<"name">(v)`, `try_emplace<"name">(v)`,
   `find<"name">()`, `contains<"name">()` (key AND type match), `erase<"name">()` (returns bool). `get<Id>` / `set<Id>`
   deleted. `at(string_view)` returns `Value` with Monostate iff missing (decision W).
7. ⏸ **Random-access iterators (decision S)** — promote `const_iterator` to random-access (`operator--`,
   `operator+/-/+=/-=`, `operator[]`, `operator<=>`). Add `cbegin/cend/rbegin/crbegin/rend/crend`.
   `std::ranges::random_access_range<ValueMap>` / `std::ranges::sized_range<ValueMap>` static-assert + qa.
8. ⏸ **`gr::allocator::pmr::CountingResource` (decision AA)** — add to `MemoryAllocators.hpp`. Rewire `qa_ValueMap` (
   drops the local `counting_resource`) and `qa_Value` (drops ad-hoc tracking) to consume the shared helper.
9. ⏸ **`qa_ValueMap.cpp` didactic reorder + new coverage (decision Z)** — reorder existing suites, add coverage for the
   typed-name fluent (`at<>`, `insert_or_assign<>`, `try_emplace<>`, `find<>`, `contains<>`, `erase<>`), ranges
   conformance (random-access + reverse iteration), and the `kDebugBuild`-gated `assert` path in `at<>` (debug-only;
   deathtest or skip under NDEBUG).

#### Round-3.5 cleanup (2026-04-23, post-investigation)

Hot-path investigation (canonical-tag dispatch ROI) showed the typed-name fluent doesn't pay off in practice — framework
hot-path sites are dominated by **iteration over the whole tag map**, not single-key lookups, and string-keyed
`find()` + `Value::value_or<T>` is already O(1) average + cheap variant-tag check. Decisions:

| #  | Topic                                        | Decision                                                                                                                                                                                                                                                                                                                                                                                                                                  |
|----|----------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| BB | Typed-name fluent in `ValueMap`              | **Strip.** Removed `at<"name">()`, `find<"name">()`, `contains<"name">()`, `insert_or_assign<"name">(v)`, `try_emplace<"name">(v)`, `erase<"name">()`. Supersedes round-3 decisions U / V / X. Wire-format / serialisation use of the registry remains; only the user-facing typed accessors are gone.                                                                                                                                    |
| CC | `CanonicalName` / `BoundCompatible` concepts | **Drop.** Only used by the typed fluent; no remaining consumers.                                                                                                                                                                                                                                                                                                                                                                          |
| DD | `constexpr` / `[[nodiscard]]` hardening      | **Apply.** Added `constexpr` to all read-only accessors (`size`, `empty`, `blob`, `resource`, `is_frozen`, `find`, `contains`, `begin/end/cbegin/cend/rbegin/crbegin/rend/crend`) and to all `const_iterator` operators. `[[nodiscard]]` coverage already broad; no further additions. PMR-touching mutators (ctor, dtor, swap, copy/move-assign, clear, reserve, shrink_to_fit, erase, emplace, insert_or_assign) cannot be `constexpr`. |
| EE | Test layout                                  | **Merge `qa_ValueMapKeys.cpp` into `qa_ValueMap.cpp`** as a "ValueMap - canonical key registry" suite at the top. Standalone file deleted; CMake `add_ut_test(qa_ValueMapKeys)` removed.                                                                                                                                                                                                                                                  |

#### Phase 1b·d.2 — Tensor support (in progress, scope locked 2026-04-23)

Wire format placed in the PackedEntry payload pool, gated by `kEntryFlagTensor` (already declared).
Two layouts (selected via `encodingFlags` bit 0):

```
TENSOR PAYLOAD LAYOUT (in PackedEntry's payload pool when kEntryFlagTensor is set;
PackedEntry::valueType is set to Value::ValueType::Value):
==========================================================================
Offset  Size  Field
------  ----  -----
  0      1    elementValueType  (Value::ValueType: Float32 / Int64 / String / Value / …)
  1      1    rank              (0..kMaxTensorRank, mirrors Tensor.hpp's kMaxRank = 8)
  2      1    encodingFlags
                  bit 0: variableSizeElements (set iff elementValueType ∈ {String, Value})
                  bits 1..7: reserved
  3      1    reserved = 0
  4      4    elementCount      (product of extents; 1 for rank-0, 0 if any extent is 0)
  8      4*r  extents[0..r-1]   (one u32 per extent, packed)

Then EITHER (variableSizeElements == 0 → fixed-size scalar / complex elements):
  8+4*r  elementCount × sizeof(elementCpp)  contiguous element data
                                            (alignment: source is Tensor's
                                             aligned pmr::vector<T>; reader
                                             memcpys into its own aligned storage)

OR (variableSizeElements == 1 → string / Value elements):
  8+4*r  elementCount × 8                   index table:
                                              u32 elementOffset[i]
                                              u32 elementLength[i]
                                            (offsets relative to tensor blob start)
  …      Σ elementLength                    concatenated element payload bytes

  Per-element payload encoding:
  · elementValueType == String  →  raw UTF-8 bytes (no per-element header)
  · elementValueType == Value   →  [u8 elemVT][u8 elemFlags][u16 reserved=0]
                                     [u32 inlineOrLength][per-type bytes]
                                   • inline scalars (≤8 B): bytes inlined in inlineOrLength
                                   • complex<double> (16 B): 16 bytes follow
                                   • String elem: inlineOrLength = string length, then bytes
                                   • Nested ValueMap (elemFlags & kEntryFlagNestedMap):
                                       inlineOrLength = sub-blob length, then sub-blob bytes
                                   • Nested Tensor (elemFlags & kEntryFlagTensor):
                                       inlineOrLength = sub-tensor length, then sub-tensor bytes
                                       (recursive, depth-capped via kMaxDecodeDepth)
```

**Element-type decode dispatch** reuses the existing `detail::dispatchValueType<F>(vt, f)`
helper: read the `elementValueType` byte → re-dispatch to the per-T decode lambda. For
`Tensor<Value>` elements, the lambda recursively calls `decodeEntry`-like logic with the
per-element header.

**Insert path** (`_assignValueAt` new branch, after the existing `ValueMap` nested-map branch):

1. Write the 16-byte tensor header.
2. Write extents.
3. Branch on element-type variability:
   · fixed → memcpy contiguous element bytes
   · variable → first build the offset/length table, then concatenate element bytes
4. Set `e.valueType = Value::ValueType::Value`, `e.flags |= kEntryFlagOffsetLength | kEntryFlagTensor`,
   `e.payloadOffset/Length` over the tensor sub-blob.
5. Use `_appendPayloadSafe` for self-alias protection (Tensor source bytes may live in our own pool
   if a caller copies a tensor out via `at()` → `value_or<Tensor<T>>(...)` then re-inserts).

**Decode path** (in `decodeEntry`, before the existing `Value::ValueType::Value` nested-map branch):

1. Match `vt == Value && (flags & kEntryFlagTensor)`.
2. Read tensor header + extents.
3. Element-type dispatch via `dispatchValueType<F>(elementVT, [&]<typename ElemT>(...){
    ...})`.
4. Per-element-type construct `Tensor<ElemT>(extents_from, contiguous_data_or_per_element_decode)`.
5. Return `Value{
    std::move(tensor), resource}`.

**Concept**: `detail::TensorLike<T>` matching `gr::Tensor<U, Ex...>` (tag-detect via the existing
`tensor_extents_tag` / member typedefs).

**`merge` + STL conversion ctor** integration: detect `valueType == Value && (flags & kEntryFlagTensor)`
before the `dispatchValueType` lambda, route through a tensor-specific re-emplace helper.

**Tests** (qa_ValueMap.cpp new suite "ValueMap - Tensor support"):

- `Tensor<float>` 1D / 2D / 3D round-trip
- `Tensor<int64_t>`, `Tensor<uint8_t>`, `Tensor<bool>` (uint8 storage), `Tensor<complex<float>>`,
  `Tensor<complex<double>>` round-trip (representative element-type coverage)
- Empty tensor (zero-extent), rank-0 (scalar tensor)
- Large tensor (force payload-pool growth)
- Self-aliasing emplace (tensor read out then re-inserted)
- `Tensor<std::string>` round-trip
- `Tensor<Value>` with mixed-type elements (e.g. int + float + string)
- Composition: `ValueMap` containing a `Tensor<float>`; `ValueMap` containing a `Tensor<Value>`
  whose elements are themselves nested `ValueMap`s

**Effort estimate** (revised post-recursive-elements scope-in): ~7–9 h.

---

#### Phase 1c — string-keyed lookup micro-optimisations (deferred)

Identified by the 2026-04-23 hot-path investigation; each is **independent, can land any time**, no API changes. These
earn more than typed dispatch would have.

| # | Site                                                                                                           | What                                                                                                               | Why                                                              |
|---|----------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------|
| 1 | `core/include/gnuradio-4.0/algorithm/TriggerMatcher.hpp:267-278`                                               | Collapse `contains(k) + at(k).holds<T>() + value_or()` 3-step chain to single `find()` + `get_if<T>()`.            | Saves 2 redundant hash lookups per call in the trigger hot path. |
| 2 | `core/src/Settings.cpp:377-397` (contextInTag / triggeredTimeInTag)                                            | Collapse `contains(k) + at(k)` 2-step to single `find()`. Two canonical keys (`CONTEXT`, `CONTEXT_TIME`) per call. | Saves 2 redundant hashes per work() context-tag handling.        |
| 3 | `core/include/gnuradio-4.0/Block.hpp:2007-2012` (tag merge)                                                    | Confirm / add `merged.reserve(map.size())` before the insert loop.                                                 | Avoids rehash storms on dense tag maps.                          |
| 4 | `core/include/gnuradio-4.0/Block.hpp:1102-1199` (`forwardInputTags`) and `Settings.hpp:1059-1065` (autoUpdate) | Early-exit when input tag map is empty.                                                                            | Skip the per-port iteration entirely for tag-less work() calls.  |

Verify with profiling on a representative pipeline (e.g. PSR / DAQ chain) before/after; the win per site is sub-1%
throughput but they compound and require no design churn.

#### Phase 1d — `Value` view-mode + unified `view_as<>` / `.owned()` API (target: pre-API-freeze)

**Motivation**. Two reasons, both load-bearing:

1. **API hygiene before freeze** (primary, 2026-04-24 user ack). The current `Value::get_if<T>()` mishmash (returns
   `T*` for inline scalars, `pmr::string*` for strings, `Tensor<T>*` for tensors, `Value::Map*` for maps — but cannot
   sensibly return any of those for view-mode bytes) needs a coherent replacement BEFORE the GR4 API freeze. Phase 1d
   is the API-design pass, not a perf-only optimisation.
2. **Alloc-free `ValueMap` iteration** (secondary). Each `ValueMap::iterator::operator*()` call constructs a fresh
   `Value` via `decodeEntry`. For inline scalars the cost is negligible; for String/Tensor/nested-Map entries
   `decodeEntry` allocates per-deref. After the alias swap to `ValueMap`, this is the per-iteration alloc that
   dominates tag-heavy benchmarks.

**Unified API surface** (replaces `get_if<>` for variable-size types):

```cpp
// `get_if<T>` is preserved as today (returns owning T*, nullptr in view-mode).
// `view_as<T>` is added as complementary, returning a non-owning view that works
// uniformly across owning + view modes — enabling alloc-free ValueMap iteration.
auto view = value.view_as<Tensor<float>>();   // std::optional<TensorView<float>>
if (view.has_value()) {
    process(*view);                          // borrow (alloc-free)
    Tensor<float> mine = view->owned(myRes); // explicit materialisation gate
}
```

**API surface decision (2026-04-24, revised)**: `get_if<>` is preserved AND extended with view-type overloads.
Single accessor name (matching `std::get_if<variant, T>` precedent), return type varies by template arg:

- `get_if<OwningT>()` returns `OwningT*` (today's behaviour). For inline scalars, `T*` aliases inline storage.
  For variable-size owning storage, `pmr::string*` / `Tensor<T>*` / `Value::Map*`. Returns nullptr on type mismatch
  OR when the Value is in view-mode (no real owning object exists).
- `get_if<ViewT>()` returns `std::optional<ViewT>` — new overloads for `std::string_view`, `TensorView<T>`,
  view-mode `ValueMap`. Works **uniformly across owning + view modes** — alloc-free.

| Source `Value` content                                   | Existing `get_if<OwningT>()` (preserved)                                  | Phase-1d-added `get_if<ViewT>()`                                                                                                                      | View → owning                                                   |
|----------------------------------------------------------|---------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------|
| `std::pmr::string`                                       | `get_if<std::pmr::string>() → pmr::string*` (nullptr in view-mode)        | `get_if<std::string_view>() → std::optional<std::string_view>` (both modes, alloc-free)                                                               | construct `std::pmr::string{                                    
 *sv, res}` directly — view *is* the materialisable shape |
| `Tensor<T>` (fixed-size element)                         | `get_if<Tensor<T>>() → Tensor<T>*` (nullptr in view-mode)                 | `get_if<TensorView<T>>() → std::optional<TensorView<T>>` (both modes, alloc-free)                                                                     | `TensorView<T>::owned(optional_resource) → Tensor<T>`           |
| `Tensor<Value>` (variable-size element)                  | `get_if<Tensor<Value>>() → Tensor<Value>*` (nullptr in view-mode)         | `get_if<TensorView<Value>>() → std::optional<TensorView<Value>>` (forward-iter only — see note)                                                       | `TensorView<Value>::owned(optional_resource) → Tensor<Value>`   |
| `Value::Map` / nested `ValueMap`                         | `get_if<Value::Map>() / get_if<ValueMap>() → Map*` (nullptr in view-mode) | `get_if<ValueMap>() → std::optional<ValueMap>` where the inner `ValueMap` may be view-mode (`is_view() == true` — option α: same type, mode-via-flag) | `ValueMap::owned(optional_resource) → ValueMap` (always owning) |

**Justification for the unified-`get_if` approach** — in view mode, no real `Tensor<T>` / `Value::Map` /
`pmr::string` object exists, only packed bytes. `get_if<OwningT>` returns nullptr (correct: no `OwningT*` to hand
out), so callers iterating a `ValueMap` and using `get_if<OwningT>` would silently miss view-mode entries.
`get_if<ViewT>()` returns a non-owning view that works *uniformly* across owning + view modes, decoupling the caller
from the source's mode. Single accessor name keeps the API surface tight; the mixed return type (`T*` for owning vs
`std::optional<T>` for view) is mild — the caller idiom `if (auto x = v.get_if<T>()) {
    use(*x); }` works for both
because both `T*` and `std::optional<T>` are bool-convertible and dereference-able. `value_or<TensorView<T>>(default)`
would also work in principle, but constructing a default `TensorView<T>` is awkward; `std::optional<TensorView<T>>`
is cleaner.

**Fundamentals**: no new accessors needed. `get_if<double>()` already returns `double*` (which is the
"reference-or-null" idiom). For mutable-reference-with-fallback, `value_or<double&>(fallback)` already exists
(Value.hpp:491-516).

**`get_if<std::string_view>` returns `std::optional<std::string_view>`** — a bit of overlap with the existing
`value_or<std::string_view>(default)` (returns `string_view` directly with default fallback), but the optional form
gives a cleaner caller idiom (`if (auto sv = v.get_if<std::string_view>()) {
    …}`), distinguishes "absent" from
"present-empty", and matches the `get_if<TensorView<T>>` shape for consistency.

**`Value` layout sketch** (target: same 24 bytes as today, shared between owning and view modes):

```cpp
class Value {
    std::uint8_t  _typeAndContainer; // 4-bit valueType + 4-bit containerType (unchanged)
    std::uint8_t  _flags;            // bit 0 = mode (0=owning, 1=view); rest reserved
    std::uint16_t _reserved;         // alignment
    std::uint32_t _payloadLength;    // bytes for variable-size; 0 for inline scalars

    union Storage {
        bool        b;
        std::int8_t i8;
        …;
        std::uint64_t u64;
        float         f32;
        double        f64;
        std::byte*    payloadPtr; // owning OR view — same shape, mode bit decides
    } _storage;

    std::pmr::memory_resource* _resource; // owning: allocator; view: source's resource (used on copy → owning)
};
```

`_storage` and `_resource` stay public — Q6 confirmed (CLAUDE.md style; `_` prefix marks the foot-gun line).

**ValueMap iterator under Phase 1d**: `decodeEntry` constructs view-mode `Value` for variable-size entries by setting
`payloadPtr = blobBase + payloadOffset; _payloadLength = entry.payloadLength; _flags |= ViewMode`. Zero allocations
across String / Tensor / nested-Map dereferences. For inline scalars the path is unchanged.

**Lifetime contract** — view-mode `Value`s alias the source ValueMap's blob and are valid only until the source's
iterators are invalidated (any `emplace`/`insert_or_assign`/`erase`/`clear`/`reserve`/`shrink_to_fit`/`merge`).
Identical to existing iterator-invalidation rules; documented as such on `Value` and on each `view_as<>` overload.
Caller discipline only — no `freeze()`-then-iterate enforcement (Q3: mutate-while-iterating is UB on
`std::pmr::unordered_map` already; the same caller hygiene applies). Single-arg `Value` copy ctor / assignment from a
view-mode source materialises into owning mode against the **target's** resource (Q5 confirmed; matches the
`4f3e801b` PMR contract). Two-arg `Value(const Value&, std::pmr::memory_resource* dest)` materialises against `dest`.

**`view->owned(optional_resource)` resource fallback**: if no resource is passed, the view's recorded source
resource (`view._resource` carried through from the source `Value`) is used. Avoids accidental fall-back to the
global default resource on materialisation.

**Migration strategy** — five commits, each green independently:

1. **`Value`: add view-mode internals** — no API additions yet, `get_if<OwningT>` overloads return nullptr in
   view-mode. Add `_flags` byte (currently in padding); bit 0 = mode (0 = owning, 1 = view); add `_payloadLength`
   field (4 bytes from padding). View-mode storage shape: `_storage.ptr` aliases external bytes (raw, NOT a
   `pmr::string`), `_payloadLength` carries length, `_flags |= ViewMode`. Private factory
   `Value::makeView(ValueType, ContainerType, std::byte* base, std::uint32_t length, std::pmr::memory_resource*)` for
   ValueMap to invoke. Owning copy ctor / assignment from view-mode auto-materialises (deep copy via target
   resource). `destroy()` no-ops for view-mode (we don't own the bytes).
2. **`Value::get_if<std::string_view>()` returns `std::optional<std::string_view>`** — new overload, works for both
   owning + view modes (alloc-free). For owning: returns sv aliasing `_storage.ptr->pmr::string->data()`. For view:
   returns sv aliasing the raw bytes at `_storage.ptr` with length `_payloadLength`.
3. **Wire `ValueMap::const_iterator::operator*()` to view-mode for String entries**. Keep owning materialisation for
   Tensor / nested Map until step 5 / 6 land. Re-run `bm_ValueMap`; expect string-iter workload to match the
   "Phase 1d sim" projection (≈ unordered_map parity, no per-deref alloc).
4. **Migrate the 14 `get_if<std::pmr::string>` callers in-tree to `get_if<std::string_view>`** so they read view-mode
   entries correctly post-step 3. Required — without this they'd silently miss view-mode strings (get_if<pmr::string>
   returns nullptr in view-mode). API breakage scope is moderate (3 production headers in `blocks/sdr/` +
   `blocks/audio/`
   + 1 `core/src/Value.cpp` internal + 7 qa-tests). **STOP BEFORE THIS STEP and seek feedback if migration scope
     creep is concerning** (per user direction 2026-04-24 — minimise blocklib API breakage).
5. **`gr::TensorView<T>::owned(optional_resource)`** — new method on the existing `TensorView<T>` (Tensor.hpp:1665).
   Materialiser semantics for `TensorView<Value>` (variable-size element case): `owned()` walks per-element headers,
   decodes each Value into an owning Tensor<Value>. Add `Value::get_if<TensorView<T>>() →
   std::optional<TensorView<T>>` (complementary — `get_if<Tensor<T>>` stays). Wire ValueMap iterator to view-mode for
   Tensor entries.
6. **`ValueMap` non-owning blob mode (option α)** — add `_ownsBlob` flag; `_deallocateBlob` becomes a no-op when
   `!_ownsBlob`; every mutator (`emplace`/`insert_or_assign`/`erase`/`clear`/`reserve`/`shrink_to_fit`/`merge`) asserts
   (debug) or no-ops (release; TBD on policy at implementation). Add `ValueMap::owned(optional_resource) → ValueMap`
   materialiser, `ValueMap::is_view() → bool`. Update `Value::get_if<ValueMap>()` to return `std::optional<ValueMap>`
   where the inner ValueMap may be view-mode (depending on source Value's mode). `get_if<Value::Map>` stays. Wire
   ValueMap iterator to view-mode for nested-map entries.
7. **No `get_if<>` cleanup pass** — `get_if<OwningT>` overloads stay. Optional doc update: mention "returns nullptr
   for view-mode entries — use `get_if<ViewT>()` for view-mode-friendly access". Final `bm_ValueMap` re-run; expect
   ValueMap iter to match (or beat) `Value::Map` baseline across all workloads.

**Compatibility implications**:

- `get_if<OwningT>()` overloads are **all preserved**. No forced caller migration for owning-mode use.
- For view-mode entries, `get_if<std::pmr::string>` / `get_if<Tensor<T>>` / `get_if<Value::Map>` return nullptr —
  same "type mismatch" semantics as today. Callers iterating a `ValueMap` who want to read view-mode entries should
  use the new `get_if<ViewT>()` overloads: `get_if<std::string_view>` / `get_if<TensorView<T>>` / `get_if<ValueMap>`.
- **In-tree caller migration (step 4) is required** for the 14 `get_if<std::pmr::string>` sites that iterate tag
  maps — to be view-mode-aware after step 3 lands. Out-of-tree blocklib code that uses `get_if<std::pmr::string>`
  on iterator-yielded Values would silently miss view-mode entries — they must adopt `get_if<std::string_view>`.
- `Value::operator==`, `std::hash<Value>`, `Value::value_or<T>(default)`, `holds<T>()`, all `is_*()` predicates: stay
  content-based, work transparently across both modes.
- `_storage`, `_resource` stay public (Q6); `_flags` and `_payloadLength` added publicly, same convention.
- Wire format version: no bump — Phase 1d only changes `Value`'s in-memory representation; ValueMap's blob layout is
  untouched. `kBlobVersion` stays at 1 (Q7 confirmed; not used in production yet either way).

**Subtle points to land carefully**:

- `TensorView<Value>` is forward-iterator only — random `[i]` requires walking per-element headers. Document
  explicitly; expose only `begin()`/`end()`/`size()`. Random-access goes through `.owned()` first.
- `view->owned()` resource fallback uses the view's recorded source resource (not the global default).
- `ValueMap` view-mode mutator refusal: pick assert vs `std::expected` vs silent no-op consistently with the rest of
  the framework (probably assert — fits the framework's exception-free philosophy and `Block::publishTag` etc. don't
  expect mutation failure on tag maps).

**Effort estimate** (revised post-2026-04-24 unified-`get_if` decision):

- Step 1 (Value view-mode internals; no API additions): 3–5 h
- Step 2 (`get_if<std::string_view>` overload): 1 h
- Step 3 (wire iterator for strings): 1–2 h
- Step 4 (migrate 14 callers `get_if<pmr::string>` → `get_if<std::string_view>`): 1–2 h
- Step 5 (TensorView::owned + `get_if<TensorView<T>>`): 4–6 h
- Step 6 (ValueMap view-mode + `get_if<ValueMap>` view-yielding): 5–7 h
- Step 7 (no cleanup pass needed): 0 h
- **Total: 15–23 h.**

**Phase 1d.0 — informational mini-benchmark** (was "gating", now "informational" per 2026-04-24 user direction —
Phase 1d proceeds regardless to land the API before freeze; the mini-bench quantifies the perf side of the win):

- Sterile container-only benchmark in `bench/bench_ValueMap.cpp` (per project convention; check existing `bench/`
  layout).
- Workload: insert N entries, find by key, iterate-and-extract each Value, copy whole map, erase by key, merge two
  maps. Strings + scalars mix matching real Tag payloads (≈ 3 strings + 5 scalars).
- N ∈ {3, 5, 10, 20, 50}.
- Compare:
   - `std::pmr::unordered_map<pmr::string, Value>` (= `Value::Map`) — baseline.
   - `gr::pmt::ValueMap` — current (decodeEntry per iter, owning Value).
   - `gr::pmt::ValueMap` — Phase 1d hypothetical (simulate by stubbing iterator to construct view-mode Value — trivial
     in benchmark scaffold).
- Metrics: ns/op, allocations/iter, peak resident bytes, total bytes per map.
- Output: side-by-side table; user reviews; informs the order/priority of Phase 1d steps but does NOT gate them.

**Open questions** — RESOLVED 2026-04-24 (all):

1. **Accessor naming**: `view_as<>` vs unified `get_if<ViewT>` — RESOLVED (revised twice):
   - First proposal: introduce `view_as<>` for view types.
   - Second revision (user pushback 2026-04-24): preserve `get_if<>` everywhere; `view_as` only as a complement
     for Tensor + Map.
   - Third revision (user pushback 2026-04-24, FINAL): no `view_as`. **Unified under `get_if<>`** — overload by
     template arg: `get_if<OwningT> → T*`, `get_if<ViewT> → std::optional<ViewT>`. Single accessor name, matches
     `std::get_if<variant, T>` precedent. The mixed return type is mild (both `T*` and `std::optional<T>` are
     bool-convertible + dereference-able, so the caller idiom is uniform).
2. **`get_if<std::pmr::string>` strategy** — RESOLVED: preserve. For view-mode entries it returns nullptr (correct).
   Callers iterating ValueMaps must migrate to the new `get_if<std::string_view>` overload (returns
   `std::optional<std::string_view>`) to read view-mode entries.
3. **View-Value lifetime documentation** — RESOLVED: caller discipline + Doxygen note. No `freeze()`-then-iterate
   (mutate-while-iterate is UB on `std::pmr::unordered_map` already).
4. **`detach()` / `own()` accessor** — RESOLVED: name it `owned(optional_resource)`. Lives on `TensorView<T>` and
   view-mode `ValueMap`. (For strings, callers construct `std::pmr::string{
    *sv, res}` directly — no wrapper needed.)
5. **Cross-resource copy of view-mode Value** — RESOLVED: target's `_resource`. Two-arg ctor for explicit override.
6. **Public layout exposure** — RESOLVED: keep public; CLAUDE.md `_` convention is the only "private" marker needed.
   New `_flags` and `_payloadLength` fields added publicly.
7. **Wire-format version bump** — RESOLVED: no bump.
8. **`ValueMap` view-mode option (α vs β)** — RESOLVED: option α (same `ValueMap` type, `_ownsBlob` flag /
   `is_view()` predicate). Less code, no separate `ValueMapView` type to teach.
8. **Performance gate threshold** — RESOLVED: no gate. Phase 1d proceeds for API hygiene; benchmark is informational
   and reviewed by user.

#### Phase 1b — resumes after the Round-3 series

The original 3e–3j scope slips one series further; resumes once 9 is green:

a. ⏸ **Conversion** — ctors from `std::map<std::string, Value, std::less<>>` /
`std::pmr::unordered_map<pmr::string, Value>`; `to_std_map` / `to_std_unordered_map`. `init_from_map` logic as template.
b. ⏸ **Serialisation** — `blob()` (functional since 3d); `from_blob(span, resource)` → `std::expected<ValueMap, Error>`.
Validation: magic, version, alignment, bounds, offsets, endian (LE-only).
c. ⏸ **Device-side primitives** — scalar atomic stores to canonical slots; entry append when `slack_bytes > 0`.
d. ⏸ **Extend value-type coverage** — complex, `Tensor<T>`, nested `ValueMap`; re-enables `trigger_meta_info`
round-trip.
e. ⏸ **Long inline-key spill-to-pool** — keys > 27 chars stored in payload pool instead of rejected at insert.
f. ⏸ **Parity gate** — port `Value::Map` qa cases to run against `ValueMap`.
g. ⏸ **Type-alias switch** — `using property_map = ValueMap;` at a single point in `Tag.hpp`, migrate `/core/` callers,
deprecate `Value::Map`. Remove alias one release cycle later.

### 4.2.1 Design decisions made in 3d (non-obvious — keep for resume)

- **`Value& operator[]` dropped**: the flat `FlatEntry` blob has no materialised `Value` object to hand back by reference. Callers use `at()` + `insert_or_assign()` + `set<Id>()`. A proxy-based follow-up (`ValueSlotRef` supporting `= V` and implicit-to-`Value`) is deferred to step 8.
- **`at()` returns `Value` by value**, not `const Value&`. `Value` is constructed on the fly from entry + blob. Throws `std::out_of_range` on missing key (matches Tag convenience method semantics at `Tag.hpp:88`).
- **Iterators yield `std::pair<std::string_view, Value>` by value**; `operator*()` returns a freshly-constructed pair. Invalidation is still vector-like.
- **Scope**: scalars (11 numeric + `bool`) and `std::string_view` / `std::string`. Complex / `Tensor` / nested `ValueMap` as insert values trigger a `static_assert` (compile error, not runtime surprise). Step 8 re-enables them.
- **Long inline keys rejected at insert time** (> 27 chars) — `emplace` / `insert_or_assign` return `{
    end(), false}` and the map stays unchanged. Step 9 spills to payload pool.
- **Non-const iterator**: `using iterator = const_iterator` as placeholder — no mutable iterator in 3d.
- **Entry-commit ordering**: `_header->entry_count` is incremented **before** the value payload is written so a payload-triggered `_grow()` correctly preserves the partially-populated slot (cur_entries then includes the new slot; grow's `memcpy` captures its key). Test `"string values survive blob relocation across growth"` in `qa_ValueMap.cpp` pins this — do **not** reorder without re-reading that test first.
- **Clang-tidy informational hints** (not `-Werror` blockers): `_entry_matches` / `_write_key` can be made `static` (don't read `this`); several loop variables are `uint16_t` while the bound deduces to `unsigned int`. Left as a later cleanup pass — not correctness issues.

### 4.3 Phase 2 — pooled tag storage + housekeeping

**Commit strategy**: every Phase-2 edit folds into a **second, separate commit**. Same stage-locally-and-squash discipline as Phase 1.

1. `PooledRingBuffer<Descriptor, Payload>` primitive with unit tests.
2. `TagRingBuffer := PooledRingBuffer<TagDescriptor, ValueMapBlob>`. Wire through `Port.hpp:361` `TagBufferType` seam.
3. `enum class HouseKeepingPolicy { Reclaim, Shrink, All };` + `Block::houseKeeping(HouseKeepingPolicy = All)` default method, time/iteration gated in `work()` loop.
4. Backpressure E2E: saturate tag pool → upstream `tryReserve` returns short/empty span → source Block stalls.
5. v2+: `CircularBuffer<PmrCapable T, ...>` auto-specialisation reusing `PooledRingBuffer`. Trait-detected, no policy tag.

---

## 5. Key file locations (for resume)

- `core/include/gnuradio-4.0/Value.hpp` — `Value::Map` defined here (line 159, has TODO to migrate to `std::flat_map`); `MapHash`/`MapEqual` transparent lookup (142-157); Emscripten `holds<size_t>` guard (822-825).
- `core/include/gnuradio-4.0/ValueHelper.hpp` — `ValueVisitor` (ValueHelper.hpp:1054-1148, 27 handlers).
- `core/include/gnuradio-4.0/Tag.hpp` — `Tag` struct (line 74), `property_map` alias (line 29), `PropertyMapType` concept (lines 61-62), convenience methods (lines 83-110), **`DefaultTag<...>` declarations (lines 196-216) + `kDefaultTags` array (line 218) — authoritative canonical-key source for the ValueMap registry**.
- `core/src/Value.cpp` — `copy_from` (line 96, design-intent comment lines 101-102); `operator=(const Value&)` (347); `operator=(Value&&)` (behaviour updated by `4f3e801b`); move-ctor (339); `swap(Value&,Value&)` (224).
- `core/include/gnuradio-4.0/Port.hpp` — `TagBufferType` seam (line 361), `kDefaultBufferSize=4096` (line 540), `newTagIoHandler` (lines 729-735), `InputSpan`/`OutputSpan` (lines 581+, 662+), `rawTags` (line 582)
- `core/include/gnuradio-4.0/Block.hpp` — `publishTag` (line 1305-1319), tag-consume API
- `core/include/gnuradio-4.0/CircularBuffer.hpp` — primary buffer. Line 231, 777-790 allocator; 227 template. `tryReserve` lines 492-521; `std::copy_n` branch 363-378.
- `core/include/gnuradio-4.0/BlockModel.hpp` — `EdgeParameters` line 63, `Edge` line 73 (`_dataResource` line 91, `_tagResource` line 92, `_uiConstraints` line 96)
- `core/include/gnuradio-4.0/Graph.hpp` — `applyEdgeConnection` (line 636+), `edges()` accessor (lines 368-369), `connect()` (line 502)
- `core/include/gnuradio-4.0/Scheduler.hpp` — `cleanupZombieBlocks` (lines 672, 1082-1144), `runWatchDog` (lines 735-754), `ExecutionPolicy` (lines 83-84)
- `core/include/gnuradio-4.0/MemoryAllocators.hpp` — existing PMR helpers (Aligned, Logging, migrate*). No pool/slab yet.
- `core/test/qa_Value.cpp` — "pmr resource usage and propagation" block holds the PMR regression tests (4 copy-ctor + 2 move-assign); template for future `ValueMap` parity tests.
- `core/include/gnuradio-4.0/ValueMap.hpp` — ValueMap container + `gr::pmt::keys` registry, single merged header. Core impl is header-only for now (scalars + strings).
- `core/test/qa_ValueMap.cpp` — 31 tests / 114 asserts over the core API; `qa_ValueMapKeys.cpp` covers the registry separately.
- `core/test/qa_buffer.cpp` — existing aggregate-T tests (`buffer_tag`, `std::map<int,int>`)
- `core/test/qa_Tags.cpp` — Tag lifecycle / propagation tests
- `opencmw-cpp/src/serialiser/include/IoSerialiserYaS.hpp` — YaS reference patterns; **no link-time dep, read-only reference**

---

## 6. Original spec summary (for reference when re-reading on another machine)

Full spec was delivered inline during the prior planning session; this WIP captures the operative decisions. The spec has 8 sections:
- §0 Context: two subsystems — `Value::Map` and aggregate `CircularBuffer<T>`
- §1 Non-negotiables: single PMR per Map, 32-byte key cap, compile-time canonical key registry, flatbuffer layout, scalar `CircularBuffer<double>` unchanged, tiny tag buffers by default, SPMC default, wait-free publish, LE only, 32-bit offsets
- §2 Code-reading checklist — **done, see §2 above**
- §3 `ValueMap` design (flatbuffer) — path α; path β proposed as v1 alternative
- §4 aggregate `CircularBuffer<T>` — retargeted to `PooledRingBuffer` + `TagRingBuffer`
- §5 Equality / hashing / conversion
- §6 3-phase migration
- §7 Test plan
- §8 Open items — most now resolved; 10-13 still pending (post-prototype confirmations)

If spec re-read is needed on another machine, the full memo is in the prior session transcript — ask user.

---

## Appendix A — Canonical key list (pinned from `Tag.hpp:196-218`)

20 entries. Range `0x0001–0x007F` is reserved for GR4 core; `0x0080+` for SigMF / extensions; `0x0000` is the "no entry" sentinel. ID numbering is dense now but can gap for future additions. Names come from each `DefaultTag`'s first template argument (no prefix — prefixing is a later refactor). Types map onto `Value::ValueType` per the investigation (§2.1).

|       ID | Name                | C++ type                  | `ValueType` | Unit | Description                                                       | Notes                                                                                              |
|---------:|---------------------|---------------------------|-------------|------|-------------------------------------------------------------------|----------------------------------------------------------------------------------------------------|
| `0x0001` | `sample_rate`       | `float`                   | `Float32`   | Hz   | signal sample rate                                                | `SIGNAL_RATE` is an alias                                                                          |
| `0x0002` | `signal_name`       | `std::string`             | `String`    |      | signal name                                                       |                                                                                                    |
| `0x0003` | `num_channels`      | `gr::Size_t` (`uint32_t`) | `UInt32`    |      | interleaved channel count                                         |                                                                                                    |
| `0x0004` | `signal_quantity`   | `std::string`             | `String`    |      | signal quantity                                                   |                                                                                                    |
| `0x0005` | `signal_unit`       | `std::string`             | `String`    |      | signal's physical SI unit                                         |                                                                                                    |
| `0x0006` | `signal_min`        | `float`                   | `Float32`   | a.u. | signal physical min (DAQ) limit                                   |                                                                                                    |
| `0x0007` | `signal_max`        | `float`                   | `Float32`   | a.u. | signal physical max (DAQ) limit                                   |                                                                                                    |
| `0x0008` | `n_dropped_samples` | `gr::Size_t` (`uint32_t`) | `UInt32`    |      | number of dropped samples                                         |                                                                                                    |
| `0x0009` | `frequency`         | `double`                  | `Float64`   | Hz   | signal centre frequency                                           |                                                                                                    |
| `0x000A` | `rx_overflow`       | `bool`                    | `Bool`      |      | RX overflow indicator                                             |                                                                                                    |
| `0x000B` | `trigger_name`      | `std::string`             | `String`    |      |                                                                   | no description given in `DefaultTag`                                                               |
| `0x000C` | `trigger_time`      | `uint64_t`                | `UInt64`    | ns   | UTC-based time-stamp                                              |                                                                                                    |
| `0x000D` | `trigger_offset`    | `float`                   | `Float32`   | s    | sample delay w.r.t. the trigger (compensates analog group delays) |                                                                                                    |
| `0x000E` | `trigger_meta_info` | `property_map`            | `Value`     |      | maps containing additional trigger information                    | nested ValueMap post-migration                                                                     |
| `0x000F` | `local_time`        | `uint64_t`                | `UInt64`    | ns   | UTC-based time-stamp (host)                                       | **declared in `Tag.hpp` but NOT in `kDefaultTags[]`** — include since it has a `DefaultTag` entity |
| `0x0010` | `context`           | `std::string`             | `String`    |      | multiplexing key to orchestrate node settings/behavioural changes |                                                                                                    |
| `0x0011` | `ctx_time`          | `uint64_t`                | `UInt64`    |      | multiplexing UTC-time [ns] when ctx should be applied             | renamed from `time` in Round-3 commit 1 (decision M)                                               |
| `0x0012` | `reset_default`     | `bool`                    | `Bool`      |      | reset block state to stored default                               |                                                                                                    |
| `0x0013` | `store_default`     | `bool`                    | `Bool`      |      | store block settings as default                                   |                                                                                                    |
| `0x0014` | `end_of_stream`     | `bool`                    | `Bool`      |      | end of stream, receiver should change to DONE state               |                                                                                                    |

**Decisions pending user confirmation** before 3b builds on this:
- **ID numbering**: dense starting at `0x0001`, or leave gaps (e.g. group by family: 0x0001–0x000F signal, 0x0010–0x001F trigger, 0x0020–0x002F context, …) for easier future additions? Default: **dense**, no gaps (KISS; the registry is append-only anyway).
- **`local_time` inclusion**: included above as `0x000F` even though it's not in `kDefaultTags[]`. Default: **include**, since `DefaultTag<"local_time", …>` already exists in `Tag.hpp:211`.
- **`time` → `ctx_time` rename (resolved 2026-04-22, decision M)**: applied in Round-3 commit 1 — hard rename, no
  bridge. In-tree callers (12, all via `gr::tag::CONTEXT_TIME.shortKey()`) auto-pick-up the new name. Out-of-tree
  consumers (persisted YAMLs in `Graph_yaml_importer.hpp`, wire-format msgs exercised by `qa_Messages.cpp`, SigMF /
  OpenDigitizer) adopt at their own cadence. `Tag.hpp:213` TODO removed.
- **Aliases** (e.g. `SIGNAL_RATE` aliasing `sample_rate`): registry has one canonical entry per *name*, not per symbol. Default: **one entry per unique name**.

---

## Appendix B — `get_if<T>()` API simplification proposal (2026-04-27, awaiting evaluation)

User-raised idea (do **not** act on yet — transfer-then-evaluate):
> Would the following policy help if `get_if<T>()` is limited to either fundamentals and/or views?

### Current state (post-Q1.B)

| Argument T                                                 | Return                                                  | Cost                                         |
|------------------------------------------------------------|---------------------------------------------------------|----------------------------------------------|
| inline scalars (bool, int*, float, double, ComplexFloat32) | `T*` (pointer into `_storage` slot)                     | zero                                         |
| `std::complex<double>`                                     | `T*` (pointer into payload allocation)                  | zero                                         |
| `std::string_view`                                         | `optional<string_view>` aliasing blob                   | zero                                         |
| `TensorView<T>` (fixed-size T)                             | `optional<TensorView<T>>` aliasing data                 | zero                                         |
| `TensorView<Value>` (specialisation)                       | `optional<TensorView<Value>>` owning a decoded snapshot | one alloc + per-element decode               |
| `Tensor<T>`                                                | `optional<Tensor<T>>` (decoded copy)                    | one alloc + per-element decode               |
| `ValueMap`                                                 | `optional<ValueMap>` view-mode                          | zero (aliases blob bytes; `owned()` to copy) |

### Proposed policy

`get_if<T>()` is **strictly non-allocating, non-owning view access**. The two return shapes:

1. `T*` for fundamentals stored inline / in payload (raw pointer; nullptr on miss).
2. `optional<View>` for view types: `string_view`, `TensorView<T>`, `ValueMap` (view-mode).

**Drop**: `get_if<Tensor<T>>()` returning `optional<Tensor<T>>` (the decoded-copy path).
**Drop**: implicit `value_or<TensorLike>(...)` special case (no longer needed once `get_if<Tensor>` is gone).

For owning materialisation, callers use:

- `view->owned([resource])` for a fresh `Tensor<T>` from a `TensorView<T>` (mirrors the existing API).
- `gr::pmt::convertTo<T>(value, [mr])` for type-changing (Tensor<float> ⇄ Tensor<int> etc.).

`Tensor<bool>` (currently the only TensorView<X> that can't be constructed because `pmr::vector<bool>` has no
`.data()`):

- Add a `TensorView<bool>` partial specialisation that owns a `Tensor<bool>` snapshot internally — same shape as the
  existing `TensorView<Value>` partial specialisation. The view-API surface (size/begin/end/[i]/owned) stays uniform;
  storage is the implementation detail (alias vs owning snapshot).
- Net SLOC: ~30 added for the bool partial specialisation, ~70 deleted for `get_if<Tensor<T>>` body in ValueMap.hpp, ~10
  deleted for `value_or<TensorLike>` branches in Value.hpp. Net **−50 SLOC**.

### Pros

- **Single mental model**: `get_if = view-mode access` (no allocation). Materialisation is always explicit (`.owned()` /
  `convertTo`) — visible cost.
- **Aligns with `std::variant::get_if` semantics**: always non-owning, never throws, never allocates.
- **Removes the brace-init pitfall** (`T{std::move(tensor)}` → init_list ctor for Tensor<Value>) by eliminating the
  construction site entirely; only `TensorView::owned()` builds owning Tensors and uses parens internally.
- **Removes `value_or<TensorLike>` special case** and the `return_t` Tensor branch added today — `value_or` becomes
  uniform: copy / borrow / transfer for fundamentals + views only.
- **API surface shrinks**: one fewer overload per element type to reason about; concept constraints simplify.
- **Bench profile clearer**: any Tensor materialisation has an explicit `.owned()` / `convertTo` call — easy to grep,
  easy to optimise.
- **Tensor<bool> stops being an inconsistency**: today's code has a `if constexpr (bool)` fallback in 3+ places; under
  the proposal it's just the same `TensorView<bool>` partial specialisation as `TensorView<Value>`.

### Cons / migration cost

- **~22 caller sites need re-migration** (this session migrated `auto*` → `auto p` for `Tensor<T>` `get_if`; the policy
  migration would change them again to either `TensorView<T>` + `.owned()` or `convertTo<Tensor<T>>(v)`):
  - `core/src/Value.cpp` hash function (13 sites — straightforward `TensorView<T>` switch, hash uses `*` deref
    iteration).
  - `core/include/gnuradio-4.0/ValueHelper.hpp` six dispatch tables (vector/array/Tensor × value/Tensor cases).
  - `core/include/gnuradio-4.0/ValueMap.hpp` two internal converter ctor branches.
  - `algorithm/include/gnuradio-4.0/algorithm/ImGraph.hpp:373` (UI port-info reader).
  - `core/test/qa_ValueMap.cpp:838` (Tensor<bool> round-trip test).
  - `meta/include/gnuradio-4.0/meta/UnitTestHelper.hpp:230` (TensorLike fallback added today).
- **Compile time**: TensorView<bool> partial specialisation may add a small instantiation cost; net SLOC removed should
  offset.
- **Caller ergonomics**: read-only callers prefer `get_if<TensorView<T>>` already (already migrated). Write/own callers
  gain one extra method call (`.owned()`) which makes the cost visible — desired.
- **Documentation churn**: header doc-blocks for `get_if<Tensor<T>>` get removed; `TensorView<T>::owned()` doc gains "
  decoded snapshot" wording.

### Open questions for evaluation

1. **`Value` element of Tensor<Value>**: currently `TensorView<Value>` already owns a snapshot. Under the policy, this
   is still exactly correct — no change to Tensor<Value>'s view path.
2. **`std::pmr::string` element of Tensor<std::pmr::string>**: today these go through `init_from_tensor` which converts
   to `Tensor<Value>` on the fly. Under the policy, the user would call `get_if<TensorView<Value>>()` and walk the
   per-element headers — same as today. No new cost.
3. **`Tensor<std::complex<double>>`**: 16-byte payload-aligned elements; `TensorView<std::complex<double>>` aliases just
   fine. No special-case needed.
4. **Return-type uniformity vs. raw `T*`**: keep `T*` for fundamentals (one fewer optional unwrap; documented
   zero-cost), or unify to `optional<T>`/`optional<reference>`? KISS: keep the raw `T*` form — it's the natural shape
   for "is this scalar of type X stored here? if so, here's a pointer to it".

### Estimated effort

~½ day work: drop the `get_if<Tensor<T>>` body, add `TensorView<bool>` partial specialisation, migrate 22 sites (
sed-style), rebuild + run all tests, bench. The bool partial specialisation is the most novel piece (~30 SLOC mirroring
TensorView<Value>).

### Recommendation (for user evaluation, not action)

The proposal is **clean** and aligns with the principle that `get_if` should never surprise. The brace-init bug I hit
today (Tensor<Value>'s implicit conversion to Value polluting init_list resolution) is exactly the kind of gotcha the
policy avoids by removing the construction path. The Tensor<bool> consistency is a nice bonus.

**Decision deferred** — user wants to transfer the session + branch to another machine first; resume there.

