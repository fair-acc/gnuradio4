@group(0) @binding(0) var<storage, read_write> heap: array<atomic<u32>>;
fn word(i: u32) -> u32 { return atomicLoad(&heap[i]); }
fn setWord(i: u32, v: u32) { atomicStore(&heap[i], v); }
fn setLane(i: u32, mask: u32, bits: u32) { atomicAnd(&heap[i], ~mask); atomicOr(&heap[i], bits); }
// wasm's compare-exchange is strong; WGSL's may fail spuriously, leaving old_value == expected
fn gr_cmpxchg(i: u32, expected: u32, replacement: u32) -> u32 {
    var old = expected;
    loop {
        let r = atomicCompareExchangeWeak(&heap[i], expected, replacement);
        old = r.old_value;
        if (r.exchanged || old != expected) { break; }
    }
    return old;
}
