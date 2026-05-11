# Canonical element wire-format (V1)

A self-describing element frame shared by `gr::pmt::Value` / `ValueMap` /
`ValueView` (each value-record) and `gr::ByteRingBuffer` (each ring record): a
walker skips and type-dispatches any element from the first 8 bytes alone.

## Fixed prefix (8 bytes, little-endian)

```
[0] u32 size           element start → next element start; == capacity
[4] u8  valueType      gr::pmt::ValueType; 0xFF = escape (type in typeName)
[5] u8  containerType  Scalar | Complex | String | Tensor | Map | …
[6] u8  flags          kFlagReadOnly | kFlagChecksumPresent | kFlagAnnotationsPresent | …
[7] u8  payloadOffset  payload start from element start (== 8 when no optional region)
[8] …   optional region (payloadOffset − 8 bytes; only when present)
[po]…   payload        length is type-derived (below), not stored
```

`size == capacity`; bytes between the content and `size` are unused slack. No per-element
version or endianness byte (those are container/transport invariants; big-endian consumers
byte-swap on read).

### Flag bits

| bit  | name                      | meaning                                                                    |
| ---- | ------------------------- | -------------------------------------------------------------------------- |
| 0x01 | `kFlagReadOnly`           | payload is logically immutable; mutators MUST refuse                       |
| 0x02 | `kFlagChecksumPresent`    | optional region carries a CRC32C checksum of payload                       |
| 0x04 | `kFlagAnnotationsPresent` | optional region carries the annotation block (quantity, unit, description) |

Unknown future flag bits: writers MUST write 0, readers MUST preserve on round-trip and
ignore semantically (forward-compat).

## Optional region

Each sub-block is independently present per its own gate, written in this fixed order:

| sub-block     | gate                              | bytes                                                                 |
| ------------- | --------------------------------- | --------------------------------------------------------------------- |
| `typeName`    | `valueType == 0xFF`               | `u8 len, bytes (UTF-8), '\0'`                                         |
| `annotations` | `flags & kFlagAnnotationsPresent` | `u8 q_len, q (UTF-8); u8 u_len, u (UTF-8); u16 d_len (LE), d (UTF-8)` |
| `checksum`    | `flags & kFlagChecksumPresent`    | `u32 CRC32C of payload (LE)`                                          |

`payloadOffset = kPrefixBytes (8) + sum of present sub-block sizes`. Old readers locate
payload via `payloadOffset` even if future sub-blocks are appended here.

### Annotation block (SI quantity, SI unit, description)

`typeName` names the C++ type when escape; the annotation block describes the _physical
meaning_ of the value: an **SI quantity** (e.g. `"frequency"`, `"voltage"`), an **SI
unit** (e.g. `"Hz"`, `"m/s²"`), and a free-form **description**. All three are UTF-8 and
each is independently absent via `len == 0` (so unit-only / description-only / all-three
all encode without extra flag bits).

Field width rationale: quantity / unit are short canonical names → `u8 len` (max 255 B);
description may carry a sentence → `u16 len` (max 65 535 B). UTF-8 covers the Greek
letters and special characters required for SI (Ω, μ, π, °).

**Semantic constraint** — `kFlagAnnotationsPresent` MUST only be set when the Value is the
keyed entry of a Map. Standalone Values MUST NOT set it; receivers MAY ignore stray flags.

**Persistence convention** — an absent annotation block on a keyed Value does NOT mean
"unannotated"; the receiver MUST reuse the last block received for that key, or treat as
unannotated if none has ever been received. Producer cadence (first message / periodic /
on-change) is a caller-API concern, not a wire-format concern.

### Checksum (per-element)

`u32 CRC32C` over the payload bytes only. Useful for partial-recovery from damaged files
(skip a damaged element, continue with the rest). The per-frame CRC (below) is the
network-loss-detection layer; per-element checksum is independent and orthogonal.

## Content length (derived, never stored)

- Scalar / Complex — fixed from `valueType`.
- String — payload `[bytes][\0]`; length is the guard-`\0` offset.
- Tensor — payload opens with the tensor sub-blob descriptor (rank, extents,
  element `valueType`).
- Map — payload is a nested blob with its own container `Header`.

## Traversal

`next = h + load_u32(h+0)` (skip; payload / typeName / annotations / checksum never read).
Dispatch on `containerType` / `valueType`; resolve `typeName` only on escape. Payload at
`h + h[7]`. `size` is a multiple of the container alignment, so successive prefixes stay
aligned.

## Alignment — per-container, not a prefix property

- `Value` / `ValueMap` value-records: 16 B alignment / 16 B minimum (covers
  `std::complex<double>` and the `alignas(8)` `Header` / `PackedEntry`; >8 B SIMD
  alignment for tensor data is the caller's concern).
- `ByteRingBuffer` records: `kCacheLine` slots (MIMO false-sharing / GPU bulk);
  the embedded element still uses the 8-byte prefix.

## Type taxonomy

`valueType` = element / scalar type, `containerType` = structure (orthogonal):
`Tensor<double>` = `Float64`/`Tensor`; a nested map = `Value`/`Map`. Higher-level
aggregates (`Tag`, `Message`, `DataSet<T>`, `Packet<T>`) are not wire types — they are
compositions of the primitives; identify them on the wire via the escape `typeName`,
keeping the core enum primitive.

---

# Streaming transport — Stream header + Frame envelope (opt-in)

The per-element format above is sufficient for in-memory transports (PMR pool, USM ring)
where record boundaries are implicit in the container. For **file / network IO** an
external sender API may opt in to two extra layers: a one-time stream header and per-frame
envelopes. These are configured by the caller; in-memory paths skip them entirely.

## Stream header (8 B, once per stream/file)

```
[0] u8[4]  magic    = 'G','R','4','W'
[4] u16    version  = 0x0001     (LE; monotonic; V1 covers all of the above)
[6] u16    flags               (LE)
```

### Stream-header flag bits

| bit    | name          | meaning                                                      |
| ------ | ------------- | ------------------------------------------------------------ |
| 0x0001 | per-frame CRC | per-frame CRC present; receiver MUST verify each frame's CRC |

rest reserved (writer MUST write 0, reader MUST ignore — forward-compat).

The stream header lets `file(1)`-style tools recognise the format and lets receivers
negotiate the protocol version and frame-CRC policy.

## Frame envelope (16 B per chunk)

Each frame wraps a concatenation of self-delimited per-element records (no further
intra-frame framing needed — records carry their own size).

```
[0]  u32  frame_size  (LE)   — payload bytes only (envelope NOT included);
                                next_frame = pos + 16 + frame_size
[4]  u64  sequence    (LE)   — per-frame monotonic counter; resets to 0 on
                                each new stream header
[12] u32  crc32c      (LE)   — CRC32C of (envelope-with-this-field-zeroed || payload).
                                Receiver verifies iff stream flag 0x0001 set;
                                otherwise writer MUST write 0 and receiver MUST ignore.
```

### CRC computation (per frame, zero-out trick)

1. Producer serialises envelope with `crc32c = 0` and writes payload.
2. Producer computes `CRC32C(envelope_bytes_with_zeroed_field || payload_bytes)`.
3. Producer back-patches `crc32c`.

Receiver: read envelope, zero the CRC field locally, recompute, compare; on mismatch,
fail the frame (skip via `frame_size`).

### Per-element vs per-frame integrity

The two CRC layers are independent and orthogonal:

- **per-element** (`kFlagChecksumPresent`): partial-recovery from damaged files (skip a
  bad element, continue with the rest);
- **per-frame** (envelope CRC): network drop / framing-loss detection (whole-frame validity).

Both, either, or neither may be enabled. Some transports without an envelope (e.g. raw
USM/PMR memory) still benefit from per-element checksums.
