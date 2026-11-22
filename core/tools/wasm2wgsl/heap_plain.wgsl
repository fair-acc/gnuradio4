@group(0) @binding(0) var<storage, read_write> heap: array<u32>;
fn word(i: u32) -> u32 { return heap[i]; }
fn setWord(i: u32, v: u32) { heap[i] = v; }
fn setLane(i: u32, mask: u32, bits: u32) { heap[i] = (heap[i] & ~mask) | bits; }
