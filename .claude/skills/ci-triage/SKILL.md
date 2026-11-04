---
name: ci-triage
description: Fetch and classify a failed GitHub Actions job for this repo — distinguish out-of-memory kills, six-hour timeouts, configure errors and genuine test failures, and identify what was in flight. Use whenever a CI job fails.
---

# CI triage

## Fetching logs

`gh run view --log` refuses while the run is still in progress. This always works:

```bash
JID=$(gh run view <run-id> --json jobs -q '.jobs[]|select(.name=="<job>")|.databaseId')
gh api --allow-escape-sequences "repos/fair-acc/gnuradio4/actions/jobs/$JID/logs" \
  | sed 's/\x1b\[[0-9;]*[a-zA-Z]//g' > job.log
```

Without `--allow-escape-sequences` the call fails; without the `sed` the log is unreadable.

## Classification

| signature                                                            | cause                                             |
| -------------------------------------------------------------------- | ------------------------------------------------- |
| `exit code 137` + "runner has received a shutdown signal"            | OOM — the kernel killed the runner                |
| `Build` step with **no conclusion** at all                           | OOM so hard the step never reported               |
| conclusion `cancelled` at ~360 min                                   | six-hour job timeout                              |
| `CMake Error: generator : Ninja / Does not match ... Unix Makefiles` | a cached `_deps` sub-build from another generator |
| `***Failed <n> sec`                                                  | the test's own bound, not a ctest timeout         |
| `***Timeout`                                                         | ctest `TIMEOUT` property                          |

## After an OOM

The runner is 4 cores / ~16 GB. Read the last twenty lines: the compiles and links in flight are the
peak. Sum their measured peak RSS (see the `measure-build-cost` skill) before changing
`GR_BUILD_PARALLEL_LEVEL` — a job count sized from core count rather than memory is how this repo
OOMs.

Also grep `cores,` for the runner's own report of cores and MemTotal, and `Hits:` for the ccache hit
rate, which explains most duration differences between runs.
