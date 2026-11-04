---
name: measure-build-cost
description: Measure the peak memory and wall time of a single translation unit or link step, so build parallelism can be sized from evidence. Use before changing GR_BUILD_PARALLEL_LEVEL or when diagnosing an out-of-memory build.
---

# Measure build cost

Memory, not core count, bounds this build: single translation units have exceeded 7 GB.

## Recipe

```bash
cmake -S . -B <tree> -G Ninja -DCMAKE_BUILD_TYPE=<type> -DCMAKE_CXX_COMPILER=<cxx> [flags]
cd <tree>
o=$(grep -o "[^ ]*/<name>\.cpp\.o" build.ninja | head -1)
ninja -j4 "$o"                                     # build dependencies FIRST
rm -f "$o"
CCACHE_DISABLE=1 /usr/bin/time -f "%M %e" ninja -j1 "$o"   # KB peak RSS, seconds
```

## Why each step

**Pre-build the dependencies.** `/usr/bin/time` on a `ninja` invocation charges the whole dependency
chain to the target. Skipping this produced a 149 s figure for a 21 s translation unit and a 2.7 GB
figure for a 0.2 GB link — both wrong, both used to justify wrong conclusions.

**`CCACHE_DISABLE=1`**, or a cache hit reports the cost of copying a file.

**Delete only the target**, never the whole tree.

## Flags that change the answer

Match the CI configuration exactly. `-DADDRESS_SANITIZER=ON` adds 12-21 % peak RSS; coverage
instrumentation adds its own. A measurement without the sanitiser understates every GCC matrix job.

## Sizing

Sum the peak RSS of the translation units that appear together in the failing log, add the
shared-library link (heavy: ~3 GB; executable links are ~0.2 GB), and compare against ~16 GB. Set
`-DGR_BUILD_PARALLEL_LEVEL=<N>` from that sum.
