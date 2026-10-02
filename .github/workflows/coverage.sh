#!/usr/bin/env bash
# Clang source-based coverage for CI: coverage.sh configure | build | test | report | history | pages
#
# report merges the profiles and writes coverage-report.txt, coverage.lcov, coverage.txt (Sonar) and coverage-html/,
# compares against the default branch's last report, computes coverage of the lines changed since the merge-base,
# keeps one sticky PR comment and posts the coverage/patch status. history appends a row to history.csv on the
# coverage-data branch for default-branch pushes; pages assembles the GitHub Pages site. Base, status and history
# are best effort and never fail the job.
set -euo pipefail

SRC=$PWD
BUILD=$(realpath -m "${BUILD_DIR:-$SRC/../build}")
LLVM=${LLVM_VERSION:-20}
HEAD_SHA=${HEAD_SHA:-${GITHUB_SHA:-$(git -C "$SRC" rev-parse HEAD)}}
BASE_REF=${BASE_REF:-main}
PROFILES=$BUILD/profiles
IGNORE='(/_deps/|/test/|/usr/|/build/|/third_party/|/blocks/testing/)'
AREAS='core blocks algorithm meta'
PATCH_MIN=${COVERAGE_PATCH_MIN:-80}
PATCH_GATE=${COVERAGE_PATCH_GATE:-}
HISTORY_BRANCH=coverage-data
HISTORY_REMOTE=${COVERAGE_HISTORY_REMOTE:-origin}
METHOD="llvm-cov-$LLVM/1" # bump when tooling, flags or exclusions change, so the history marks the discontinuity
MARKER='<!-- gr4-coverage -->'
REPOSITORY=${GITHUB_REPOSITORY:-fair-acc/gnuradio4}
export GH_REPO=${GH_REPO:-$REPOSITORY}
PAGES_URL="https://${REPOSITORY%%/*}.github.io/${REPOSITORY#*/}/coverage"
INSTRUMENT='-fprofile-instr-generate -fcoverage-mapping -mllvm -runtime-counter-relocation -Wno-unused-command-line-argument'

configure() {
  cmake -S "$SRC" -B "$BUILD" -G Ninja -DCMAKE_C_COMPILER="clang-$LLVM" -DCMAKE_CXX_COMPILER="clang++-$LLVM" \
    -DCMAKE_BUILD_TYPE=Debug -DGR_BUILD_PARALLEL_LEVEL=3 -DUSE_CCACHE=ON -DCMAKE_COLOR_DIAGNOSTICS=ON \
    -DDISABLE_EXTERNAL_DEPS_WARNINGS=ON -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DENABLE_COVERAGE=OFF -DADDRESS_SANITIZER=OFF \
    -DCMAKE_C_FLAGS="$INSTRUMENT" -DCMAKE_CXX_FLAGS="$INSTRUMENT"
}

build() {
  echo "$(nproc) cores, $(awk '/MemTotal/{printf "%d", $2/1024}' /proc/meminfo) MB RAM"
  cmake --build "$BUILD"
}

run_tests() {
  mkdir -p "$PROFILES"
  export LLVM_PROFILE_FILE="$PROFILES/%c%m-%p.profraw" # %c: counters are mmapped, so a suite run in a static destructor still counts
  cd "$BUILD"
  ctest --output-on-failure || ctest --rerun-failed --repeat until-pass:3 --output-on-failure || echo "::warning::tests still fail after retries; coverage is reported regardless, the build jobs gate on tests"
}

readable() { # clang-20 emits a coverage mapping for some translation units that llvm-cov rejects; one such object aborts the whole report
  local error
  if error=$("llvm-cov-$LLVM" report -instr-profile "$BUILD/coverage.profdata" "$1" 2>&1 >/dev/null); then
    echo "$1"
  elif [[ $error != *"no coverage data found"* ]]; then
    echo "$1" >>"$BUILD/coverage-skipped.txt"
  fi
}

select_objects() {
  rm -f "$BUILD/coverage-skipped.txt"
  export -f readable
  export BUILD LLVM
  {
    find "$BUILD" -type f -perm -u+x -name 'qa_*' -not -name '*.*'
    find "$BUILD" -type f -name '*.so*' -not -path '*/_deps/*'
  } | sort -u | xargs -P "$(nproc)" -I{} bash -c 'readable "$1"' _ {} | sort >"$BUILD/coverage-objects.txt"
}

cov() {
  local verb=$1
  shift
  mapfile -t objs < <(awk 'NR > 1 { print "-object" } { print }' "$BUILD/coverage-objects.txt")
  "llvm-cov-$LLVM" "$verb" -instr-profile "$BUILD/coverage.profdata" "$@" "${objs[@]}"
}

total() { [[ -f $1 ]] && awk -v c="$2" '/^TOTAL/ && NF >= c { gsub("%", "", $c); print $c }' "$1"; }

skipped_count() { [[ -s $BUILD/coverage-skipped.txt ]] && wc -l <"$BUILD/coverage-skipped.txt" || echo 0; }

area_lines() { # per area and in total: "<area> <lines found> <lines hit>" from an LCOV file
  [[ -f $1 ]] || return 0
  awk -v areas="$AREAS" -v src="$SRC" '
    BEGIN { n = split(areas, list, " "); for (i = 1; i <= n; ++i) known[list[i]] = 1 }
    /^SF:/ { path = substr($0, 4); if (index(path, src "/") == 1) path = substr(path, length(src) + 2); area = substr(path, 1, index(path, "/") - 1); if (!(area in known)) area = "other" }
    /^LF:/ { found[area] += substr($0, 4); found["total"] += substr($0, 4) }
    /^LH:/ { hit[area] += substr($0, 4); hit["total"] += substr($0, 4) }
    END { for (a in found) print a, found[a], hit[a] + 0 }' "$1"
}

pct() { awk -v hit="$1" -v found="$2" 'BEGIN { if (found > 0) printf "%.1f", 100 * hit / found }'; }

delta() { awk -v head="$1" -v base="$2" 'BEGIN { d = head - base; if (d > -0.05 && d < 0.05) print "±0.0"; else printf "%+.1f", d }'; }

status_icon() { awk -v p="$1" -v min="$PATCH_MIN" 'BEGIN { print (p >= 90 ? "🟢" : (p >= min ? "🟡" : "🔴")) }'; }

base_report() { # the newest coverage-report artifact of the default branch; other jobs of that run may have failed
  local run
  run=$(gh api "repos/$REPOSITORY/actions/artifacts?name=coverage-report&per_page=50" \
    -q "[.artifacts[] | select(.workflow_run.head_branch == \"$BASE_REF\" and .expired == false)][0] | \"\\(.workflow_run.id) \\(.workflow_run.head_sha)\"" 2>/dev/null) || return 1
  [[ -n $run ]] || return 1
  rm -rf "$BUILD/base"
  gh run download "${run%% *}" -n coverage-report -D "$BUILD/base" >/dev/null 2>&1 || return 1
  [[ -f $BUILD/base/coverage-report.txt ]] || return 1
  echo "${run#* }" >"$BUILD/base/sha"
}

changed_lines() { # tab-separated: kind (C covered, U uncovered, N no data), area, file, line; for lines added since the merge-base
  local since
  since=${COVERAGE_DIFF_BASE:-$(git -C "$SRC" merge-base "origin/$BASE_REF" HEAD 2>/dev/null || true)}
  [[ -n $since ]] || return 0
  git -C "$SRC" diff --no-color --unified=0 "$since" HEAD -- '*.hpp' '*.cpp' '*.h' '*.hh' '*.cc' '*.cxx' |
    awk -v lcov="$BUILD/coverage.lcov" -v src="$SRC" -v ignore="$IGNORE" -v areas="$AREAS" '
      BEGIN {
        n = split(areas, list, " ")
        while ((getline record < lcov) > 0) {
          if (record ~ /^SF:/) { file = substr(record, 4); if (index(file, src "/") == 1) file = substr(file, length(src) + 2) }
          else if (record ~ /^DA:/) { split(substr(record, 4), da, ","); count[file, da[1] + 0] = da[2] + 0 }
        }
      }
      function areaOf(path,    i, a) { a = "other"; for (i = 1; i <= n; ++i) if (index(path, list[i] "/") == 1) a = list[i]; return a }
      function lineText(path, number,    l, k) {
        if (!(path in loaded)) { loaded[path] = 1; k = 0; while ((getline l < (src "/" path)) > 0) { k++; text[path, k] = l }; close(src "/" path) }
        return text[path, number]
      }
      function executableLooking(t) {
        gsub(/^[ \t]+|[ \t]+$/, "", t)
        if (t == "" || t ~ /^\/\// || t ~ /^\/?\*/ || t ~ /^#/ || t ~ /^[{}();,]*$/ || t ~ /^}[ ]*\/\// || t ~ /^(public|private|protected):$/) return 0
        return t !~ /^(template[ ]*<|using |struct |class |enum |namespace |friend |static_assert|requires|concept |typename |(gr::)?(Annotated|A)<|(gr::)?(PortIn|PortOut|MsgPortIn|MsgPortOut)<|GR_MAKE_REFLECTABLE|GR_REGISTER_BLOCK)/
      }
      /^\+\+\+ / { path = substr($0, 5); sub(/^b\//, "", path); skip = (path == "/dev/null" || ("/" path) ~ ignore); next }
      /^@@ / && !skip {
        split($3, plus, ","); first = substr(plus[1], 2) + 0; howMany = (plus[2] == "" ? 1 : plus[2] + 0)
        for (line = first; line < first + howMany; ++line) {
          if ((path, line) in count) print (count[path, line] > 0 ? "C" : "U") "\t" areaOf(path) "\t" path "\t" line
          else if (executableLooking(lineText(path, line))) print "N\t" areaOf(path) "\t" path "\t" line
        }
      }'
}

line_ranges() { # "file L1, L3–L5" per file from tab-separated "file line"; at most $1 entries, then "+ N more"
  awk -F'\t' -v cap="$1" '
    function flush() { if (file != "") entries[++n] = "- `" file "` " ranges; ranges = "" }
    function add(a, b) { ranges = ranges (ranges == "" ? "" : ", ") "L" a (b > a ? "–L" b : "") }
    $1 != file { if (file != "") add(start, last); flush(); file = $1; start = last = $2; next }
    $2 == last + 1 { last = $2; next }
    { add(start, last); start = last = $2 }
    END { if (file != "") { add(start, last); flush() }; for (i = 1; i <= n && i <= cap; ++i) print entries[i]; if (n > cap) print "- + " n - cap " more files, full list in the artifact" }'
}

render() {
  local head=$BUILD/coverage-report.txt base="" baseLcov="" baseSha=""
  if base_report; then
    base=$BUILD/base/coverage-report.txt
    baseSha=$(<"$BUILD/base/sha")
    [[ -f $BUILD/base/coverage.lcov ]] && baseLcov=$BUILD/base/coverage.lcov
  fi
  changed_lines >"$BUILD/coverage-changed.tsv"

  local changedTotal changedCovered patchPct headLines baseLines
  changedTotal=$(awk -F'\t' '$1 != "N"' "$BUILD/coverage-changed.tsv" | wc -l)
  changedCovered=$(awk -F'\t' '$1 == "C"' "$BUILD/coverage-changed.tsv" | wc -l)
  patchPct=$(pct "$changedCovered" "$changedTotal")
  headLines=$(area_lines "$BUILD/coverage.lcov" | awk '$1 == "total" && $2 > 0 { printf "%.1f", 100 * $3 / $2 }')
  baseLines=$([[ -n $baseLcov ]] && area_lines "$baseLcov" | awk '$1 == "total" && $2 > 0 { printf "%.1f", 100 * $3 / $2 }' || true)
  echo "$changedTotal $changedCovered ${patchPct:-}" >"$BUILD/coverage-patch.txt"

  local mode="informational"
  [[ -n $PATCH_GATE ]] && mode="gate"
  echo "$MARKER"
  if ((changedTotal > 0)); then
    echo "### $(status_icon "$patchPct") Coverage · \`${HEAD_SHA:0:8}\`${baseSha:+ vs \`$BASE_REF\` @ \`${baseSha:0:8}\`}"
    echo
    echo "**Changed code: $patchPct % of $changedTotal lines covered** ($mode ${PATCH_GATE:-$PATCH_MIN} %) · project lines **$headLines %${baseLines:+ ($(delta "$headLines" "$baseLines"))}**"
  else
    echo "### Coverage · \`${HEAD_SHA:0:8}\`${baseSha:+ vs \`$BASE_REF\` @ \`${baseSha:0:8}\`}"
    echo
    echo "**No changed executable lines** · project lines **$headLines %${baseLines:+ ($(delta "$headLines" "$baseLines"))}**"
  fi
  echo
  echo '| area | lines | Δ vs base | changed lines | covered |'
  echo '|---|---:|---:|---:|---:|'
  local area found hit bfound bhit headPct basePct changed covered
  for area in $AREAS other total; do
    found="" hit="" bfound="" bhit=""
    read -r found hit < <(area_lines "$BUILD/coverage.lcov" | awk -v a="$area" '$1 == a { print $2, $3 }') || true
    [[ -n ${found:-} && $found -gt 0 ]] || continue
    headPct=$(pct "$hit" "$found")
    basePct=""
    if [[ -n $baseLcov ]]; then
      read -r bfound bhit < <(area_lines "$baseLcov" | awk -v a="$area" '$1 == a { print $2, $3 }') || true
      [[ -n ${bfound:-} && $bfound -gt 0 ]] && basePct=$(pct "$bhit" "$bfound")
    fi
    if [[ $area == total ]]; then
      changed=$changedTotal
      covered=$changedCovered
    else
      changed=$(awk -F'\t' -v a="$area" '$1 != "N" && $2 == a' "$BUILD/coverage-changed.tsv" | wc -l)
      covered=$(awk -F'\t' -v a="$area" '$1 == "C" && $2 == a' "$BUILD/coverage-changed.tsv" | wc -l)
    fi
    local name="\`$area/\`" changedCell="–" coveredCell="–"
    [[ $area == other ]] && name="other"
    [[ $area == total ]] && name="**total**"
    if ((changed > 0)); then
      changedCell=$changed
      coveredCell="$(status_icon "$(pct "$covered" "$changed")") $(pct "$covered" "$changed") %"
    fi
    local deltaCell="n/a"
    [[ -n $basePct ]] && deltaCell=$(delta "$headPct" "$basePct")
    printf '| %s | %s %% | %s | %s | %s |\n' "$name" "$headPct" "$deltaCell" "$changedCell" "$coveredCell"
  done

  local uncovered noData
  uncovered=$(awk -F'\t' '$1 == "U" { print $3 "\t" $4 }' "$BUILD/coverage-changed.tsv")
  noData=$(awk -F'\t' '$1 == "N" { print $3 "\t" $4 }' "$BUILD/coverage-changed.tsv")
  if [[ -n $uncovered ]]; then
    echo
    echo "<details><summary>⚠️ $(wc -l <<<"$uncovered") changed lines not covered</summary>"
    echo
    line_ranges 20 <<<"$uncovered"
    echo
    echo '</details>'
  fi
  if [[ -n $noData ]]; then
    echo
    echo "<details><summary>❔ $(wc -l <<<"$noData") changed lines without coverage data (e.g. templates no test instantiates; not counted)</summary>"
    echo
    line_ranges 20 <<<"$noData"
    echo
    echo '</details>'
  fi

  echo
  echo '<details><summary>regions, functions, branches, instantiations</summary>'
  echo
  echo '| | base | head | Δ |'
  echo '|---|---:|---:|---:|'
  local label column h b
  for label in "regions 4" "functions 7" "branches 16" "instantiations *(informational)* 10"; do
    column=${label##* }
    label=${label% *}
    h=$(total "$head" "$column")
    b=$([[ -n $base ]] && total "$base" "$column" || true)
    if [[ -n $b ]]; then
      printf '| %s | %s %% | %s %% | %s |\n' "$label" "$b" "$h" "$(delta "$h" "$b")"
    else
      printf '| %s | n/a | %s %% | n/a |\n' "$label" "$h"
    fi
  done
  echo
  echo '</details>'
  echo
  local skipped=""
  [[ -s $BUILD/coverage-skipped.txt ]] && skipped=" · not counted, unreadable coverage mapping: $(xargs -n1 basename <"$BUILD/coverage-skipped.txt" | sort | sed 's/.*/`&`/' | paste -sd' ')"
  echo "🟢 ≥ 90 % · 🟡 ≥ $PATCH_MIN % · 🔴 < $PATCH_MIN % · [artifacts](https://github.com/$REPOSITORY/actions/runs/${GITHUB_RUN_ID:-}) · [latest $BASE_REF report]($PAGES_URL/) · [history]($PAGES_URL/history.html)$skipped"
}

sticky_comment() {
  [[ -n ${PR_NUMBER:-} ]] || return 0
  local id
  id=$(gh api "repos/$REPOSITORY/issues/$PR_NUMBER/comments" --paginate --jq "[.[] | select(.body | startswith(\"$MARKER\"))][0].id // empty" | head -n 1) || return 0
  if [[ -n $id ]]; then
    gh api -X PATCH "repos/$REPOSITORY/issues/comments/$id" -F body=@"$BUILD/coverage-summary.md" >/dev/null || true
  else
    gh api -X POST "repos/$REPOSITORY/issues/$PR_NUMBER/comments" -F body=@"$BUILD/coverage-summary.md" >/dev/null || true
  fi
}

patch_status() { # coverage/patch on the PR head; fails only once COVERAGE_PATCH_GATE is set
  [[ -n ${PR_NUMBER:-} ]] || return 0
  local changedTotal changedCovered patchPct state=success description
  read -r changedTotal changedCovered patchPct <"$BUILD/coverage-patch.txt"
  if ((changedTotal == 0)); then
    description="no changed executable lines"
  else
    description="$patchPct % of $changedTotal changed lines covered"
    if [[ -n $PATCH_GATE ]]; then
      description+=" (gate $PATCH_GATE %)"
      awk -v p="$patchPct" -v g="$PATCH_GATE" 'BEGIN { exit !(p < g) }' && state=failure
    else
      description+=" (informational, target $PATCH_MIN %)"
    fi
  fi
  gh api -X POST "repos/$REPOSITORY/statuses/$HEAD_SHA" -f state="$state" -f context=coverage/patch \
    -f description="$description" -f target_url="https://github.com/$REPOSITORY/actions/runs/${GITHUB_RUN_ID:-}" >/dev/null || true
}

history_row() {
  local area found hit pr row
  pr=$(gh api "repos/$REPOSITORY/commits/$HEAD_SHA/pulls" -q '.[0].number // empty' 2>/dev/null || true)
  row="$(git -C "$SRC" log -1 --format=%cI "$HEAD_SHA"),$HEAD_SHA,${pr:-},$METHOD"
  for column in 13 4 7 16 10; do row+=",$(total "$BUILD/coverage-report.txt" "$column")"; done
  for area in $AREAS; do
    found="" hit=""
    read -r found hit < <(area_lines "$BUILD/coverage.lcov" | awk -v a="$area" '$1 == a { print $2, $3 }') || true
    row+=",$([[ -n ${found:-} && $found -gt 0 ]] && pct "$hit" "$found" || true)"
  done
  echo "$row,$(wc -l <"$BUILD/coverage-objects.txt"),$(skipped_count)"
}

record_history() { # appends one row per default-branch commit to history.csv on the coverage-data branch
  [[ ${GITHUB_EVENT_NAME:-} == push && ${GITHUB_REF:-} == "refs/heads/$BASE_REF" ]] || return 0
  local csv=$BUILD/history.csv row parent blob tree commit attempt
  row=$(history_row)
  export GIT_AUTHOR_NAME=github-actions GIT_AUTHOR_EMAIL="41898282+github-actions[bot]@users.noreply.github.com"
  export GIT_COMMITTER_NAME=$GIT_AUTHOR_NAME GIT_COMMITTER_EMAIL=$GIT_AUTHOR_EMAIL
  for attempt in 1 2 3; do
    parent=""
    if git -C "$SRC" fetch -q "$HISTORY_REMOTE" "refs/heads/$HISTORY_BRANCH" 2>/dev/null; then
      parent=$(git -C "$SRC" rev-parse FETCH_HEAD)
      git -C "$SRC" show "$parent:history.csv" >"$csv"
    else
      echo 'date,sha,pr,method,lines,regions,functions,branches,instantiations,core,blocks,algorithm,meta,binaries,skipped' >"$csv"
    fi
    grep -q ",$HEAD_SHA," "$csv" && return 0
    echo "$row" >>"$csv"
    blob=$(git -C "$SRC" hash-object -w "$csv")
    tree=$(printf '100644 blob %s\thistory.csv\n' "$blob" | git -C "$SRC" mktree)
    commit=$(git -C "$SRC" commit-tree "$tree" ${parent:+-p "$parent"} -m "coverage history: ${HEAD_SHA:0:8}")
    git -C "$SRC" push -q "$HISTORY_REMOTE" "$commit:refs/heads/$HISTORY_BRANCH" && return 0
    sleep "$attempt"
  done
  echo "coverage history: push to $HISTORY_BRANCH failed" >&2
}

pages() {
  rm -rf "$BUILD/site"
  mkdir -p "$BUILD/site"
  cp "$SRC/docs/pages/index.html" "$SRC/docs/logo-with-text.svg" "$BUILD/site/"
  cp -r "$BUILD/coverage-html" "$BUILD/site/coverage"
  cp "$SRC/docs/pages/coverage-history.html" "$BUILD/site/coverage/history.html"
  if [[ ! -f $BUILD/history.csv ]] && git -C "$SRC" fetch -q "$HISTORY_REMOTE" "refs/heads/$HISTORY_BRANCH" 2>/dev/null; then
    git -C "$SRC" show FETCH_HEAD:history.csv >"$BUILD/history.csv"
  fi
  [[ -f $BUILD/history.csv ]] && cp "$BUILD/history.csv" "$BUILD/site/coverage/history.csv"
  return 0
}

report() {
  "llvm-profdata-$LLVM" merge -sparse -failure-mode=all "$PROFILES"/*.profraw -o "$BUILD/coverage.profdata"
  select_objects
  cov report -show-instantiation-summary -ignore-filename-regex="$IGNORE" >"$BUILD/coverage-report.txt"
  cov export -format=lcov -ignore-filename-regex="$IGNORE" >"$BUILD/coverage.lcov"
  cov show -show-instantiations=false -ignore-filename-regex="$IGNORE" >"$BUILD/coverage.txt" # Sonar reads this via sonar.cfamily.llvm-cov.reportPath
  cov show -format=html -output-dir="$BUILD/coverage-html" -show-instantiations=false -ignore-filename-regex="$IGNORE" >/dev/null
  render >"$BUILD/coverage-summary.md"
  [[ -z ${GITHUB_STEP_SUMMARY:-} ]] || cat "$BUILD/coverage-summary.md" >>"$GITHUB_STEP_SUMMARY"
  sticky_comment
  patch_status
  rm -rf "$SRC/coverage-out" "$SRC/coverage-html-out"
  mkdir -p "$SRC/coverage-out" # upload-artifact takes paths inside the workspace
  cp "$BUILD"/coverage-report.txt "$BUILD"/coverage-summary.md "$BUILD"/coverage.lcov "$BUILD"/coverage-changed.tsv "$BUILD"/coverage-objects.txt "$SRC/coverage-out/"
  [[ -f $BUILD/coverage-skipped.txt ]] && cp "$BUILD/coverage-skipped.txt" "$SRC/coverage-out/"
  cp -r "$BUILD/coverage-html" "$SRC/coverage-html-out"
  cat "$BUILD/coverage-summary.md"
}

case ${1:-} in
configure) configure ;;
build) build ;;
test) run_tests ;;
report) report ;;
history) record_history ;;
pages) pages ;;
*) echo "usage: $0 configure|build|test|report|history|pages" >&2 && exit 2 ;;
esac
