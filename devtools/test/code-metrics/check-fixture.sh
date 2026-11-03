#!/usr/bin/env bash
# devtools/code-metrics against its fixture: a split function's split-invariant complexity equals clang-tidy's score of
# its hand-inlined twin, a callee with two callers or a test is not folded, and ambient reach is as listed below
# usage: check-fixture.sh <code-metrics binary> <fixture build directory>
set -euo pipefail
tool=${1:?code-metrics binary}
fixtureBuild=${2:?fixture build directory}
fixtureRoot=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$fixtureBuild/devtools"
ln -sf "$tool" "$fixtureBuild/devtools/code-metrics"
report=$(cd "$fixtureRoot" && ../../code-metrics.sh "$fixtureBuild" --root src --root test --cognitive --list --jobs 2)
failures=0
fail() {
    echo "FAIL: $*"
    failures=$((failures + 1))
}

placesOf() { # "<value>\t<key>\t<where>" of each place --list prints under the figure named $1
    awk -v figure="  $1: " '
        index($0, figure) == 1 { inside = 1; next }
        /^  [^ ]/ { inside = 0 }
        inside && /^    [0-9]+ \| / { line = substr($0, 5); first = index(line, " | "); rest = substr(line, first + 3); last = index(rest, " | ")
                                     print substr(line, 1, first - 1) "\t" substr(rest, 1, last - 1) "\t" substr(rest, last + 3) }' <<<"$report"
}
valueOf() { # the value of the place of function $2 (key "src/<file> $2(…)") under figure $1
    placesOf "$1" | awk -F'\t' -v name="$2" '$2 ~ ("^src/[^ ]+ " name "\\(") { print $1; exit }'
}
scoresOf() { # "<split-invariant> <plain>" of a function: its places under the two complexity figures, "none" for a missing one
    local splitInvariant plain
    splitInvariant=$(valueOf "split-invariant complexity" "$1")
    plain=$(valueOf "cognitive complexity" "$1")
    echo "${splitInvariant:-none} ${plain:-none}"
}
for scenario in depthZero depthTwo chain wrapper; do
    read -r splitInvariant _ <<<"$(scoresOf "${scenario}Split")"
    read -r _ inlinedPlain <<<"$(scoresOf "${scenario}Inlined")"
    [[ $splitInvariant =~ ^[0-9]+$ && $splitInvariant == "$inlinedPlain" ]] ||
        fail "$scenario: split-invariant ${splitInvariant:-none} of the split version, clang-tidy scores the inlined one ${inlinedPlain:-none}"
done
for unfolded in twoCallersFirst testedCaller; do
    read -r splitInvariant plain <<<"$(scoresOf "$unfolded")"
    [[ $plain =~ ^[0-9]+$ && $splitInvariant == "$plain" ]] || fail "$unfolded folded a callee it must not fold: ${splitInvariant:-none} against ${plain:-none}"
done
for kept in sharedPart testedPart; do
    [[ -n $(valueOf "split-invariant complexity" "$kept") ]] || fail "$kept was folded away"
done

declare -A expectedReach=(
    [drawingClock]="drawing context"
    [frameStart]="drawing context"
    [wallClock]="clock"
    [homeDirectory]="environment"
    [bumpCounter]="global"
    [nextIdentifier]="global"
    [counterPlusOne]="global"
    [counterThroughTwoCalls]="global"
    [fileExists]="io"
    [~CountOnExit]="global"
)
reachOf() { # the categories of a function's place "line <n>: <categories> (directly: …)" under the ambient-reach figure
    placesOf "functions without ambient reach" | awk -F'\t' -v name="$1" '$2 ~ ("^src/[^ ]+ (.*::)?" name "\\(") { categories = $3; sub(/^line [0-9]+: /, "", categories); sub(/ \(directly:.*$/, "", categories); print categories; exit }'
}
for function in "${!expectedReach[@]}"; do
    [[ $(reachOf "$function") == "${expectedReach[$function]}" ]] || fail "$function reaches '$(reachOf "$function")', expected '${expectedReach[$function]}'"
done
for honest in limited sortInPlace stemOf reportCount depthTwoSplit chainSplit; do
    [[ -z $(reachOf "$honest") ]] || fail "$honest reaches '$(reachOf "$honest")', expected nothing"
done

# unscoped pairs: each function's weight as counted by hand in src/Fixture.cpp, and none for a scope object or push_back
declare -A expectedPairs=(
    [pairBalanced]=1
    [pairReturnBetween]=2
    [pairLoopInside]=1
    [pairContinueSkips]=2
    [pairConditionalPush]=2
    [pairPoppedEarly]=2
)
pairWeightOf() { # the summed weight of a function's places under "unscoped pairs", keyed "src/<file>: <function>: <opener>…<closer> #<n>"
    placesOf "unscoped pairs" | awk -F'\t' -v name="$1" 'index($2, ": " name ": ") > 0 { total += $1; found = 1 } END { if (found) print total }'
}
for function in "${!expectedPairs[@]}"; do
    [[ $(pairWeightOf "$function") == "${expectedPairs[$function]}" ]] || fail "$function weighs '$(pairWeightOf "$function")' as an unscoped pair, expected ${expectedPairs[$function]}"
done
for scoped in pairScoped ScopedItem '~ScopedItem' pushBackOnly; do
    [[ -z $(pairWeightOf "$scoped") ]] || fail "$scoped counts as an unscoped pair, expected none"
done

# repeated expressions: written three times, the expression counts; its part only inside it, and an expression written twice, do not
repeatedTimes() { # how often --list says expression $1 is written in src/Fixture.cpp
    placesOf "repeated expressions" | awk -F'\t' -v text="src/Fixture.cpp: $1" '$2 == text { print $1; exit }'
}
[[ $(repeatedTimes "values.empty()?0:values.front()*2+1") == 3 ]] || fail "the expression written three times is listed '$(repeatedTimes "values.empty()?0:values.front()*2+1")' times"
[[ -z $(repeatedTimes "values.front()*2+1") ]] || fail "a part repeated only inside the larger expression is listed on its own"
[[ -z $(placesOf "repeated expressions" | grep -F '*7') ]] || fail "an expression produced by a macro invoked three times is listed"
[[ -z $(repeatedTimes "values.size()>1?values.back():0") ]] || fail "an expression written twice is listed"

# same argument everywhere: scaledBy's factor is 3 at both calls, its value varies; varyingFactor's factor varies; onceCalled has one call
constantParameter() { # the place of a parameter, "src/Fixture.cpp <signature>: <parameter>", under "same argument everywhere"
    placesOf "same argument everywhere" | awk -F'\t' -v key="$1" '$2 == key { print $3; exit }'
}
[[ $(constantParameter "src/Fixture.cpp scaledBy(int, int): factor") == "always 3, at 2 calls" ]] || fail "scaledBy's factor is '$(constantParameter "src/Fixture.cpp scaledBy(int, int): factor")'"
[[ -z $(constantParameter "src/Fixture.cpp scaledBy(int, int): value") ]] || fail "scaledBy's value varies, yet is listed"
[[ -z $(constantParameter "src/Fixture.cpp varyingFactor(int, int): factor") ]] || fail "varyingFactor's factor varies, yet is listed"
[[ -z $(constantParameter "src/Fixture.cpp onceCalled(int): factor") ]] || fail "a function called once is listed"

# write-only fields: Tally::written is assigned and initialised but never read; Tally::counted is read by +=; Compared reads all by ==
unreadField() { # the place of a field, "src/Fixture.cpp: Record::field", under "write-only fields"
    placesOf "write-only fields" | awk -F'\t' -v key="src/Fixture.cpp: $1" '$2 == key { print $3; exit }'
}
[[ -n $(unreadField Tally::written) ]] || fail "Tally::written is written and never read, yet not listed"
[[ -z $(unreadField Tally::counted) ]] || fail "Tally::counted is read by +=, yet listed"
[[ -z $(unreadField Tally::shown) ]] || fail "Tally::shown is read, yet listed"
[[ -z $(unreadField Compared::value) ]] || fail "Compared::value is read by the defaulted ==, yet listed"
[[ -z $(unreadField Dependent::seen) ]] || fail "Dependent::seen is read through a dependent type, yet listed"
[[ -n $(unreadField Dependent::unseen) ]] || fail "Dependent::unseen is only assigned, yet not listed"
[[ -n $(unreadField Tally::flags::marked) ]] || fail "an unnamed struct's field is not keyed by the member it declares: $(placesOf "write-only fields" | cut -f2 | grep marked)"

# unused functions: neverCalled is named nowhere; passedAlong is only passed as a callback, calledOnce is called, Tally's members are not functions
unusedFunction() { # the place of a function, "src/Fixture.cpp <signature>", under "unused functions"
    placesOf "unused functions" | awk -F'\t' -v key="src/Fixture.cpp $1" '$2 == key { print $3; exit }'
}
[[ -n $(unusedFunction "neverCalled(int)") ]] || fail "neverCalled is named nowhere, yet not listed"
[[ -z $(unusedFunction "passedAlong(int)") ]] || fail "passedAlong is passed as a callback, yet listed"
[[ -z $(unusedFunction "HookedBlock::settingsChanged()") ]] || fail "settingsChanged of a Block-derived class is a framework hook, yet listed"
[[ -z $(unusedFunction "HookedBlock::stop()") ]] || fail "stop of a Block-derived class is a framework hook, yet listed"
[[ -n $(unusedFunction "HookedBlock::neverCalledMethod()") ]] || fail "neverCalledMethod is no framework hook, yet not listed"
[[ -z $(unusedFunction "scaledBy(int, int)") ]] || fail "scaledBy is called, yet listed"

# flag arguments: drawn's paint is a literal at both calls; namedPaint passes a named local, which is not a literal
flagArgument() { # the place of a parameter, "src/Fixture.cpp <signature>: <parameter>", under "flag arguments"
    placesOf "flag arguments" | awk -F'\t' -v key="$1" '$2 == key { print $3; exit }'
}
[[ $(flagArgument "src/Fixture.cpp drawn(int, bool): paint") == "true at 1, false at 1 of 3 calls" ]] || fail "drawn's paint is '$(flagArgument "src/Fixture.cpp drawn(int, bool): paint")'"
[[ -z $(flagArgument "src/Fixture.cpp drawn(int, bool): value") ]] || fail "drawn's value is listed as a flag"

# nested ternaries: chosenNested holds one ?: in another; chosenNamed names the inner choice; drawn holds a single ?:
[[ $(valueOf "nested ternaries" chosenNested) == 1 ]] || fail "chosenNested holds a ?: in a ?:, listed '$(valueOf "nested ternaries" chosenNested)'"
[[ -z $(valueOf "nested ternaries" chosenNamed) ]] || fail "chosenNamed names its inner choice, yet is listed"
[[ -z $(valueOf "nested ternaries" drawn) ]] || fail "drawn holds a single ?:, yet is listed"

# the other modes: every figure measured (the registry and the printers agree), one table row each, and a report
# compared with itself unchanged throughout
summary=$(cd "$fixtureRoot" && ../../code-metrics.sh "$fixtureBuild" --root src --root test --cognitive --summary --jobs 2)
figureCount=$(grep -c '^  [^ ]' <<<"$summary" || true)
[[ $figureCount == 24 ]] || fail "--summary states $figureCount figures, the registry has 24"
! grep -q '^  [^ ].*: not measured' <<<"$summary" || fail "--summary leaves a figure unmeasured: $(grep '^  [^ ].*: not measured' <<<"$summary")"
tableRows=$(cd "$fixtureRoot" && ../../code-metrics.sh "$fixtureBuild" --root src --root test --cognitive --table --jobs 2 | grep -c '^| (' || true)
[[ $tableRows == 24 ]] || fail "--table has $tableRows rows, the registry has 24"
earlierReport=$(mktemp)
trap 'rm -f "$earlierReport"' EXIT
printf '%s\n' "$summary" >"$earlierReport"
moved=$(cd "$fixtureRoot" && ../../code-metrics.sh "$fixtureBuild" --root src --root test --cognitive --compare "$earlierReport" --jobs 2 | grep -v ' unchanged$' || true)
[[ -z $moved ]] || fail "a report compared with itself is not unchanged throughout: $moved"

# a --list report kept from an "earlier" tree, edited by hand: depthTwoInlined and wallClock did not exist yet, so the
# comparison must name them as new places under the figures they made worse, with the usual remedy; the earlier totals
# follow from the current ones, so a function added to the fixture leaves this check as it is
cognitiveNow=$(awk '/^  cognitive complexity: / { print $3; exit }' <<<"$report")
depthTwoInlined=$(valueOf "cognitive complexity" depthTwoInlined)
read -r honestNow allNow < <(awk '/^  functions without ambient reach: / { gsub(/[();]/, ""); print $7, $9; exit }' <<<"$report")
ambientNow=$(awk -v honest="$honestNow" -v all="$allNow" 'BEGIN { printf "%.1f", 100 * honest / all }')
ambientEarlier=$(awk -v honest="$honestNow" -v all="$allNow" 'BEGIN { printf "%.1f", 100 * honest / (all - 1) }')
cognitiveEarlier=$((cognitiveNow - depthTwoInlined))
earlierList=$(mktemp)
trap 'rm -f "$earlierReport" "$earlierList"' EXIT
sed -e "s/^  cognitive complexity: $cognitiveNow /  cognitive complexity: $cognitiveEarlier /" -e '/ | src\/Fixture.cpp depthTwoInlined(int) | /d' \
    -e "s/^  functions without ambient reach: $ambientNow % /  functions without ambient reach: $ambientEarlier % /" -e '/ | src\/Fixture.cpp wallClock() | /d' <<<"$report" >"$earlierList"
comparison=$(cd "$fixtureRoot" && ../../code-metrics.sh "$fixtureBuild" --root src --root test --cognitive --compare "$earlierList" --jobs 2)
grep -A3 "^(2) cognitive complexity  *$cognitiveEarlier -> $cognitiveNow .* worse$" <<<"$comparison" | grep -q "^      new   src/Fixture.cpp depthTwoInlined(int): $depthTwoInlined " ||
    fail "the comparison does not name depthTwoInlined as the new place behind cognitive complexity: $comparison"
grep -A3 "^(1) functions without ambient reach  *$ambientEarlier % -> $ambientNow % .* worse$" <<<"$comparison" | grep -q '^      new   src/Fixture.cpp wallClock()   (line [0-9]*: clock' ||
    fail "the comparison does not name wallClock as the new place behind ambient reach: $comparison"
grep -A3 '^(2) cognitive complexity ' <<<"$comparison" | grep -q '^      usual way: return early' || fail "no remedy beside cognitive complexity, which moved the worse way"

if ((failures > 0)); then
    echo "$report"
    exit 1
fi
echo "code-metrics fixture: all checks passed"
