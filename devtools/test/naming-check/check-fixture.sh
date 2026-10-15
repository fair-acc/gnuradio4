#!/usr/bin/env bash
# devtools/naming-check against its fixture: every declaration marked `expect: R<n>` is reported once under that rule
# (with the proposal after `->`, or none for `-> none`), every other line of the fixture is not, and each file's
# `expect-file:` holds for its name
# usage: check-fixture.sh <naming-check binary> <fixture build directory>
set -euo pipefail
tool=${1:?naming-check binary}
fixtureBuild=${2:?fixture build directory}
fixtureRoot=$(cd "$(dirname "$0")" && pwd)
status=0
report=$("$tool" "$fixtureBuild" --root "$fixtureRoot" --std-style fixture_std --list --jobs 2) || status=$?
failures=0
fail() {
    echo "FAIL: $*"
    failures=$((failures + 1))
}
((status == 1)) || fail "naming-check exits $status, expected 1 (violations found)"

violations=$(awk '/^violations$/ { listing = 1; next } listing && /^  / { print }' <<<"$report")
rootName=$(basename "$fixtureRoot")
while IFS= read -r file; do
    relative=${file#"$fixtureRoot"/}
    where="$rootName/$relative"
    expectedFile=$(sed -n '1s/^\/\/ expect-file: \([A-Za-z0-9]*\).*/\1/p' "$file")
    reportedFile=$(grep -cF "  $where file " <<<"$violations" || true)
    if [[ $expectedFile == ok && $reportedFile != 0 ]]; then
        fail "$relative: its name is reported, expected ok"
    elif [[ $expectedFile == R* && $reportedFile != 1 ]]; then
        fail "$relative: its name is reported $reportedFile times, expected once as $expectedFile"
    fi
    lineNumber=0
    while IFS= read -r line; do
        lineNumber=$((lineNumber + 1))
        onLine=$(grep -F -e "  $where:$lineNumber:" -e "  $where:$lineNumber " <<<"$violations" || true)
        expectation=$(sed -n 's/.*\/\/ expect: \([A-Za-z0-9]*\)\( -> \([A-Za-z0-9_]*\)\)\{0,1\}.*/\1 \3/p' <<<"$line")
        read -r rule proposal <<<"$expectation"
        if [[ -z $expectation || $rule == ok ]]; then
            [[ -z $onLine ]] || fail "$relative:$lineNumber reported, expected ${rule:-nothing}: $onLine"
            continue
        fi
        count=$(grep -c "($rule: " <<<"$onLine" || true)
        [[ $count == 1 ]] || fail "$relative:$lineNumber reported $count times under $rule, expected once: ${onLine:-nothing}"
        if [[ $proposal == none ]]; then
            [[ $onLine != *" -> "* ]] || fail "$relative:$lineNumber proposes a name that clashes: $onLine"
        elif [[ -n $proposal ]]; then
            [[ $onLine == *" -> $proposal"* ]] || fail "$relative:$lineNumber should propose $proposal: ${onLine:-nothing}"
        fi
    done <"$file"
done < <(find "$fixtureRoot" -name '*.?pp' | sort)

# the other modes report the same violations: --summary their number, --table one row each, --rule only that rule
listed=$(grep -c '' <<<"$violations")
arguments=("$fixtureBuild" --root "$fixtureRoot" --std-style fixture_std --jobs 2)
summaryTotal=$("$tool" "${arguments[@]}" --summary | sed -n 's/^naming: \([0-9]*\) violations.*/\1/p') || true
[[ $summaryTotal == "$listed" ]] || fail "--summary counts ${summaryTotal:-nothing}, --list lists $listed"
tableRows=$("$tool" "${arguments[@]}" --table | grep -c '^| `') || true
[[ $tableRows == "$listed" ]] || fail "--table has $tableRows rows, --list lists $listed"
onlyR7=$("$tool" "${arguments[@]}" --table --rule R7 | grep '^| `' | awk -F' [|] ' '{ print $2 }' | sort -u) || true
[[ $onlyR7 == R7 ]] || fail "--rule R7 reports rules '$onlyR7'"

grep -q "thread_pool.hpp" <<<"$(awk '/^std-style candidates/ { listing = 1; next } listing && /^  / { print } listing && !/^  / { listing = 0 }' <<<"$report")" ||
    fail "src/thread_pool.hpp, which includes only the standard library, is not listed as a std-style candidate"

if ((failures > 0)); then
    echo "$report"
    exit 1
fi
echo "naming-check fixture: all expectations hold"
