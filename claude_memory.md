# claude_memory.md — durable LLM context for the gnuradio4 `featValueMapExtension` branch

> Format: structured prose. Markdown is the LLM's "native" durable format — readable when
> resuming, parseable when the context window reloads, and diff-friendly across sessions.
> This file is intentionally untracked (`.gitignore` won't pick it up either; treat as a
> session-local scratchpad — copy alongside the branch when transferring machines).

---

## 1 · Where I am

- **Branch**: `featValueMapExtension`, **rebased 2026-04-27 onto
  `origin/putting-the-gnu-back-into-gnuradio4`** (was previously based on `9e904b04`).
- **Top commit**: `ae73b9de WIP` — bundles Q1.B Tensor migration + Q1.A Map view-mode + recovery
  docs (claude_wip.md, claude_memory.md). Below it: `c3804402 fix(core): Value copy/move-assign
  preserve target's PMR resource`. Base: `cd230c54 Putting the GNU back into GNU Radio 4.0`.
- **Pre-rebase backup**: `refs/backup/featValueMapExtension-pre-rebase-20260427-0824` →
  `cb88d715` (in case rebase needs to be undone, `git reset --hard refs/backup/...` will restore).
- **Rebase outcome**: clean, no manual conflict resolution needed despite 3-file overlap
  (`core/include/gnuradio-4.0/Block.hpp`, `core/test/plugins/good_math_plugin.cpp`,
  `core/test/qa_ManagedSubGraph.cpp`); git's three-way merge handled all of it. 4 core qa_
  tests verified green post-rebase (Value, ValueMap, Settings, Block).
- **Working tree**: clean — only untracked `.claude/scheduled_tasks.lock` and `blocks/onnx/`
  (pre-existing, unrelated). `claude_wip.md` and `claude_memory.md` are tracked and committed.
- **User**: Ralph J. Steinhagen (`dr.steinhagen@gsi.de` — sign-off email is `r.steinhagen@gsi.de`).
- **Date of last work**: 2026-04-27.

## 2 · What's done and what's left

**Tasks complete (#20–#23, #25, #26)**:

- `Value` Tensor storage migrated from heap `Tensor<T>*` to PMR byte-blob in `_storage.ptr`.
- `get_if<Tensor<T>>()` returns `std::optional<Tensor<T>>` (decoded copy from blob).
- `get_if<TensorView<T>>()` extended for byte-blob: aliases data section for fixed-size scalars
  (zero alloc), owns a decoded snapshot for `TensorView<Value>` / `TensorView<bool>` (the
  partial specialisations live in `Value.hpp`).
- `value_or<T>` template special-cases `gr::TensorLike<Raw>` — always returns by value.
- ~22 caller sites migrated (`auto* p = .get_if<Tensor<T>>()` → `auto p = ...; *p`).
- 10 core qa_ tests green; 8 broader-tree failures all confirmed pre-existing on baseline.
- Bench re-run captured in `claude_wip.md` §0 — TC1 +24% / TC2 +24% / TC3 +27% over post-Phase-1e
  baseline.

**Pending (#24)** — ValueMap iter view-mode for Tensor + nested-Map. Currently iter ValueMap is
9.8–10.2 M ops/s vs 77 M for `unordered_map` — 7× behind because deref still allocates owning
Values for variable-size types. Fixing this is the last piece of the Q1 inversion. Estimate:
~1 day. Independent of Q1.B Tensor — pick up when the user asks.

**Deferred for evaluation (Appendix B in `claude_wip.md`)**: user-proposed policy to limit
`get_if<T>()` to fundamentals + views (drop the decoded-`Tensor<T>` path). Write-up only,
**do not act**.

## 3 · Critical knowledge (do not relearn the hard way)

### 3.1 The Tensor<Value>→Value brace-init pitfall (today's bug)

`Value` has an implicit converting ctor from any `TensorLike`. So `Tensor<Value>{...}` brace-
init prefers the `initializer_list<Value>` ctor (rule 5 in `Tensor.hpp`) over the move-ctor when
it sees `T{std::move(other_tensor_value)}`, building a 1-element tensor wrapping the moved
tensor as a single Value. **Always use parens** for moves into `Tensor<Value>`:

```cpp
return T(std::move(tensor));    // ✓ move-construct
return T{std::move(tensor)};    // ✗ init_list with one Value
```

Sites in `ValueMap.hpp::get_if<Tensor<T>>()` and `get_if<TensorView<Value>>()` use parens.
Don't "fix" them to braces.

### 3.2 Tensor byte-blob format (in Value's `_storage.ptr`)

```
offset  size  field
------  ----  ----------------------------------------------------------------------------
   0      1   elementValueType  (Value::ValueType byte)
   1      1   rank              (0 .. kMaxTensorRank == 8)
   2      1   encodingFlags     bit 0 = variableSizeElements (set iff elementVT ∈ {Value})
   3      1   reserved = 0
   4      4   elementCount      (product of extents; 0 if any extent is 0; 1 for rank-0)
   8     4*r  extents[0..r-1]   (one u32 per extent)

Then EITHER (variableSize == 0 → fixed-size scalar / complex elements):
   8+4r  elementCount × sizeof(elementCpp)  contiguous element data

OR    (variableSize == 1 → Value elements): per-element [PackedTensorElement] headers
       (16 bytes each: valueType:1 / flags:1 / reserved:2 / inlineValue:8 / payloadLength:4)
       followed by packed payload bytes.
```

Format constants live in **`Value.hpp`** (post-Q1):

- `kMaxTensorRank = 8U`
- `kMaxTensorElements = 1U << 24`
- `kTensorBlobHeaderSize = 8UZ`
- `kTensorEncodingVariableSize = 0x01`

`PackedTensorElement` is at `gr::pmt::` namespace (NOT `gr::pmt::detail::`). Don't qualify it
with `detail::`.

`kBlobAlignment = 16` (USM/SYCL requirement) — used for Tensor and Map byte-blob allocations.

### 3.3 View-mode lifetime contract

- `Value::makeView(...)` produces a Value whose `_storage.ptr` aliases external bytes.
- View-mode is **within-iter-scope only**. Copy-ctor materialises (deep copy); move-ctor
  preserves the view (cheap pointer transfer).
- `get_if<ValueMap>()` returns `std::optional<ValueMap>` in view-mode aliasing the source's
  bytes. To escape iter scope: call `.owned([resource])` for an owning copy.
- `value_or<std::string_view>(...)` returns a string_view aliasing the source's byte-blob —
  same lifetime rule. **Pattern that bites**: a temporary Value's `value_or<string_view>` dangles
  the moment the temporary dies. Always bind to lvalue first OR use `value_or<std::string>`
  (allocates).
- `Settings::get(string_key)` already routes through `at()` (owning Value) — earlier sessions
  fixed this to avoid view-mode escape. Don't revert.

### 3.4 PMR resources

- Each `Value` carries a `_resource` (defaults to `std::pmr::get_default_resource()`).
- Move-assign preserves **target's** resource (decision in commit `4f3e801b`). Don't change.
- `_entryToValue` uses default_resource fallback when source ValueMap is view-mode (avoids
  null-resource segfault when decoding nested Tensor<Value>).
- For `Tensor<Value>` decode, fill the elements vector with `Value{resource}` so each slot's
  `_resource` matches the source — the move-assign then takes the same-resource path
  (no allocation, just bit-copy + clear-source).

### 3.5 Build / parallelism rules

- **`-j6` always** (CLAUDE.md §10). Ninja `JOB_POOLS` already cap at `compile_pool=6` /
  `link_pool=2`. Always pass `-j6` explicitly anyway. Never higher — caused OOM crashes earlier
  in this session.
- Two build trees:
    - `cmake-build-debug-gcc-15/` — `-DCMAKE_BUILD_TYPE=Debug` + `-fsanitize=address` + `-fsanitize=leak`.
    - `cmake-build-release-gcc-15/` — `-DCMAKE_BUILD_TYPE=Release`. Use this for `bm_*` and `qa_PerformanceMonitor`.
- Only build directly-affected targets during refactor iteration (named qa_ targets, not the
  full tree) — full tree rebuilds the plugin libs which dominate compile time.

### 3.6 Pre-existing test failures (do NOT mistake for regressions)

Verified 2026-04-27 by running these on the stashed pre-Q1.B working tree (commit `4ee17acf`):

- `qa_SchedulerMessages`, `qa_plugins_test`, `qa_plugin_schedulers_test`, `qa_KnownSharedLibBlocks` — SEGV at
  AddressSanitizer 0x0/0x18, all in plugin/scheduler init.
- `qa_StreamToDataSet` — heap-use-after-free in `magic_enum::cmp_equal` during FunctionGenerator settings application.
- `qa_Audio` — soundio dummy backend env issue.
- `qa_RTL2832Source` — hangs waiting for hardware.
- `qa_SoapySource` — passes on re-run, transient.

If the user reports any of these failing, **first** confirm whether the failure mode matches
the baseline before assuming Q1.B-related regression.

## 4 · File map (what each modified file does post-Q1.B)

- `core/include/gnuradio-4.0/Value.hpp` — Tensor blob format constants, `TensorView<Value>` and
  `TensorView<bool>` partial specialisations, `get_if<Tensor<T>>` / `get_if<TensorView<T>>`
  declarations, `value_or<TensorLike>` branch, `return_t` special-case for TensorLike.
- `core/include/gnuradio-4.0/ValueMap.hpp` — `init_from_tensor` body (out-of-class because it
  needs `encodeTensorBlob`), `get_if<Tensor<T>>` decoded-copy body, `get_if<TensorView<Value>>`
  snapshot body, `decodeTensorBlob` / `encodeTensorBlob` / `encodeTensorElement` /
  `decodeTensorElement`. **All `T(std::move(tensor))` constructions use parens** (see §3.1).
- `core/src/Value.cpp` — `copy_from` / `destroy` / `operator==` / `hash` for Tensor case routed
  through byte-blob (`_resource->allocate(_payloadLength, kBlobAlignment)` + memcpy on copy;
  `deallocate` on destroy; memcmp on `==`).
- `core/include/gnuradio-4.0/ValueHelper.hpp` — 6 conversion dispatch tables migrated from
  `auto* t = ...` to `auto t = ...; *t`.
- `core/include/gnuradio-4.0/Settings.hpp` — `gr::TensorLike` branch in `applyStagedImpl`
  generic path; bool-fallback path consolidated through `TensorView<TTensorElem>`.
- `algorithm/include/gnuradio-4.0/algorithm/ImGraph.hpp` — port-info reader migrated.
- `core/test/qa_ValueMap.cpp` — Tensor<bool> round-trip test migrated to optional API.
- `meta/include/gnuradio-4.0/meta/UnitTestHelper.hpp` — TensorLike branch added in
  `get_value_or_fail<T>`.

## 5 · Bench numbers (post-Q1.B Tensor, 2026-04-27, Release-gcc15, best-of-3)

`bm_ValueMap` N=20 (gr::pmt::ValueMap):
| Workload | unordered_map | Value::Map | gr::pmt::ValueMap | post-Q1.A baseline | Δ |
|----------|--------------:|-----------:|------------------:|-------------------:|---|
| insert | 1.2M | 1.1M | **1.3M**          | 1.2M | +8% |
| find | 9.3M | 12.7M | 9.8M | 8.7M | +13% |
| iter | **77.1M**     | 72.6M | 10.2M ⚠ | 10.1M | +1% (gap remains) |
| copy | 5.3M | 4.0M | **14.9M**         | 14.8M | ≈ |
| merge | 476k | 461k | 399k | 358k | +11% |
| erase | 1.1M | 927k | 1.1M | 1.1M | ≈ |

`qa_PerformanceMonitor`:
| TC | Description | Baseline | post-Q1.B | Δ |
|---:|---|---:|---:|---:|
| 1 | no tags | 353 MS/s | **441 MS/s** | +25% |
| 2 | 1 tag/10k | 267 MS/s | **375 MS/s** | +40% |
| 3 | per-sample | 468 kS/s | **593–604 kS/s** | +27% |

## 6 · How to start work on Task #24 (when the user asks)

The remaining 7× iter perf gap closes with iter view-mode for Tensor + nested-Map. Today's iter
yields owning Values for variable-size types because callers' `value_or(Tensor<T>{})` and
`Tensor::data()` patterns break with view-mode source. Q1.B Tensor migration removed the
caller-side breakage (Tensor is now byte-blob); the iter site can now safely yield view-mode
for Tensor / nested-Map.

Site to change: `core/include/gnuradio-4.0/ValueMap.hpp::ValueMap::const_iterator::operator*()`
— extend the inline switch (currently handles inline scalars + String view-mode) to add
`Tensor` (use `Value::makeView(ValueType::..., ContainerType::Tensor, blob_ptr, blob_len, res)`)
and `Map` (similar). The default branch falling through to `_entryToValue` is the one that
currently allocates per deref.

Caller breakage to expect: **none** — Q1.B Tensor migration already moved `get_if<Tensor<T>>`
to the decoded-copy path, so view-mode sources work the same as owning sources for callers.
qa_ValueMap "view-mode iter" suite (already exists) will need extension to cover Tensor / Map
view-mode iter cases.

## 7 · How NOT to take a wrong turn

- Do not "simplify" `T(std::move(tensor))` to `T{...}` — see §3.1.
- Do not act on Appendix B (get_if policy) without explicit user re-confirmation.
- Do not commit on your own initiative — user initiates every commit.
- Do not run `git push --force` or `--no-verify`.
- Do not chase the 8 pre-existing failures listed in §3.6 unless the user asks.
- Do not exceed `-j6` parallelism — OOM kills the session.
- Sign-off line is `Signed-off-by: Ralph J. Steinhagen <r.steinhagen@gsi.de>` (NOT the user-
  facing email `dr.steinhagen@gsi.de`).
