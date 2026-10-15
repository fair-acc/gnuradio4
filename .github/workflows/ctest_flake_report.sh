#!/usr/bin/env bash
# Runs ctest in the current build directory and reruns failures up to three times. A test that passes only on a rerun
# does not fail the job; it is reported as a potential flake (warning annotation and job-summary entry) for follow-up.
set -uo pipefail

ctest --output-on-failure && exit 0

failedLog=Testing/Temporary/LastTestsFailed.log
firstTryFailures=()
[[ -f $failedLog ]] && mapfile -t firstTryFailures < <(cut -d: -f2- "$failedLog")

ctest --rerun-failed --repeat until-pass:3 --output-on-failure
rerunStatus=$?

stillFailing=()
((rerunStatus != 0)) && [[ -f $failedLog ]] && mapfile -t stillFailing < <(cut -d: -f2- "$failedLog")

flakes=()
for test in "${firstTryFailures[@]}"; do
  if [[ " ${stillFailing[*]} " != *" $test "* ]]; then
    flakes+=("$test")
    echo "::warning title=Potential flake::$test failed on the first try and passed on a rerun"
  fi
done

if ((${#flakes[@]} > 0)) && [[ -n ${GITHUB_STEP_SUMMARY:-} ]]; then
  {
    echo "### Potential flakes"
    echo
    echo "Failed on the first try and passed on a rerun; not blocking, to be followed up:"
    echo
    printf -- '- `%s`\n' "${flakes[@]}"
  } >>"$GITHUB_STEP_SUMMARY"
fi

exit "$rerunStatus"
