# Dependencies

Source of truth: `gr4_declare_dependency()` calls in [`CMakeLists.txt`](CMakeLists.txt) (helper module: [`cmake/Dependencies.cmake`](cmake/Dependencies.cmake)).
Drift is checked at every build by [`cmake/lint_dependencies.cmake`](cmake/lint_dependencies.cmake).

Override resolution per build: `-DGR4_DEPENDENCY_MODE={system|fetch|system-or-fetch}` (default `system-or-fetch`); per-dep `-DGR4_DEP_<NAME_UPPER>_MODE=<mode>`.

## Required

| Dependency                                          | Pinned     | Licence           | Source                               | Used by                       |
| --------------------------------------------------- | ---------- | ----------------- | ------------------------------------ | ----------------------------- |
| [vir-simd](https://github.com/mattkretz/vir-simd)   | `v0.4.4`   | LGPL-3.0-or-later | system-or-fetch                      | all modules                   |
| [magic_enum](https://github.com/Neargye/magic_enum) | `0.9.3`    | MIT               | vendored (`third_party/magic_enum/`) | all modules, removed by C++26 |
| [ExprTk](https://github.com/ArashPartow/exprtk)     | `7b993904` | MIT               | vendored (`third_party/exprtk.hpp`)  | `algorithm/`, `blocks/basic/` |

## Optional, by subcomponent

Subdirectory self-skips at configure time when its dep is missing.

| Subcomponent                 | Dependency                                         | Mode            | Pinned                               | Licence         | Distro packages                                                                                  |
| ---------------------------- | -------------------------------------------------- | --------------- | ------------------------------------ | --------------- | ------------------------------------------------------------------------------------------------ |
| `blocks/audio/` (native)     | [libsoundio](http://libsound.io)                   | system-or-fetch | `49a1f78b` (fetch fallback, patched) | MIT             | `libsoundio-dev` (Debian/Ubuntu), `libsoundio-devel` (Fedora/SUSE), `libsoundio` (Arch/Homebrew) |
| `blocks/sdr/`                | [SoapySDR](https://github.com/pothosware/SoapySDR) | system-only     | ≥ 0.8                                | BSL-1.0         | `libsoapysdr-dev` / `soapysdr-devel` / `soapysdr`                                                |
| `algorithm/` (HTTP fileio)   | [cpr](https://github.com/libcpr/cpr)               | fetch-only      | `1.14.1` (patched + ABI-shared)      | MIT             | n/a — fetch-only (see CMakeLists.txt rationale: ExactVersion policy + pre-1.12.0 distros)        |
| `algorithm/` (HTTP prereq)   | [libcurl](https://curl.se/)                        | system-only     | system                               | curl licence    | `libcurl4-openssl-dev` / `libcurl-devel` / `curl`                                                |
| `gnuradio-options` (par-STL) | [TBB](https://github.com/uxlfoundation/oneTBB)     | system-only     | ≥ 2021                               | Apache-2.0      | `libtbb-dev` / `tbb-devel` / `tbb` (gcc + `-DENABLE_TBB=ON`)                                     |
| Python integration           | [Python3](https://www.python.org/) + NumPy         | system-only     | ≥ 3.12                               | PSF-2.0 / BSD-3 | `python3.12-dev`, `python3-numpy`                                                                |

## Test- and benchmark-only

Resolved only when `ENABLE_TESTING` / `ENABLE_BENCHMARKS` is ON (defaults follow `PROJECT_IS_TOP_LEVEL`). Not linked into shipped binaries.

| Dependency                                            | Pinned                  | Licence | Used by                                                                        |
| ----------------------------------------------------- | ----------------------- | ------- | ------------------------------------------------------------------------------ |
| [Boost.UT](https://github.com/boost-ext/ut)           | `53e17f25` (2023-04-02) | BSL-1.0 | every `qa_*.cpp`, `bm_*.cpp`, `meta/UnitTestHelper.hpp`, `bench/benchmark.hpp` |
| [cpp-httplib](https://github.com/yhirose/cpp-httplib) | `v0.18.1`               | MIT     | `qa_HttpBlock`, `qa_FileIo`, `qa_WavFile`, `qa_Audio`, `qa_SubGraphAssets`     |

## Patches

Idempotent (tolerant of already-applied state). Patches apply only on the fetch path; system installs are used as-is. libsoundio uses
[`cmake/FindLibSoundIo.cmake`](cmake/FindLibSoundIo.cmake) to locate the system shared library (upstream ships neither pkg-config nor a CMake config
on every distribution) and falls back to the patched fetch when no system copy is present. cpr is fetch-only (rationale in [CMakeLists.txt](CMakeLists.txt));
its upstream `cprConfig.cmake` is installed alongside `gnuradio4Config.cmake` so downstream `find_dependency(cpr CONFIG)` resolves.

| Patch                                                                          | Applied to | Purpose                                                 |
| ------------------------------------------------------------------------------ | ---------- | ------------------------------------------------------- |
| [`patches/libsoundio-cmake4.diff`](patches/libsoundio-cmake4.diff)             | libsoundio | CMake 4 compatibility                                   |
| [`patches/cpr-disable-std-fs-test.diff`](patches/cpr-disable-std-fs-test.diff) | cpr        | disable `std::filesystem` test broken on libstdc++ < 17 |

Refresh vendored snapshots: `./third_party/download_external_deps.sh`.

Licences are SPDX shortcodes; verify against each upstream `LICENSE` file before redistribution.
