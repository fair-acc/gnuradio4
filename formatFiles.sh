#!/usr/bin/env bash
# Restyled's formatters at the versions pinned below, over the files changed since HEAD, since <ref>, or all (--all, as CI)
# usage: ./formatFiles.sh [--since <ref> | --all] | --install [<dir>] | --versions
set -euo pipefail

kClangFormat=18.1.8
kCmakeFormat=0.6.13
kPrettier=3.8.1
kShellcheck=0.11.0
kBlack=26.1.0

cd "$(dirname "$0")"
formatters=${GR_FORMATTERS_HOME:-$HOME/.cache/gr4-formatters}
export PATH="$formatters/venv/bin:$formatters/node/node_modules/.bin:$formatters/bin:$PATH"

install() { # the pinned versions into $formatters: Python packages in a venv, prettier from npm, shellcheck's release
    mkdir -p "$formatters/bin"
    python3 -m venv "$formatters/venv"
    "$formatters/venv/bin/pip" install --quiet "clang-format==$kClangFormat" "cmakelang[YAML]==$kCmakeFormat" "black==$kBlack"
    npm install --silent --prefix "$formatters/node" "prettier@$kPrettier"
    curl --proto '=https' --tlsv1.2 -fsSL "https://github.com/koalaman/shellcheck/releases/download/v$kShellcheck/shellcheck-v$kShellcheck.linux.x86_64.tar.xz" |
        tar -xJ -C "$formatters/bin" --strip-components=1 "shellcheck-v$kShellcheck/shellcheck"
}

versions() {
    echo "clang-format $(clang-format --version 2>/dev/null | grep -o '[0-9][0-9.]*' | head -1) (pinned $kClangFormat)"
    echo "cmake-format $(cmake-format --version 2>/dev/null) (pinned $kCmakeFormat)"
    echo "prettier $(prettier --version 2>/dev/null) (pinned $kPrettier)"
    echo "shellcheck $(shellcheck --version 2>/dev/null | sed -n 's/^version: //p') (pinned $kShellcheck)"
    echo "black $(black --version 2>/dev/null | grep -o '[0-9][0-9.]*' | head -1) (pinned $kBlack)"
}

case ${1:-} in
--install)
    formatters=${2:-$formatters}
    install
    exit
    ;;
--versions)
    versions
    exit
    ;;
esac

changedOnly=true
since=HEAD
case ${1:-} in
--all) changedOnly=false ;;
--since) since=${2:?a git ref, e.g. origin/main} ;;
esac

collect() { # <pathspec>...: the tracked files to format; vendored code, patches and links are left as they are
    local file
    if [[ $changedOnly == true ]]; then
        git diff --name-only --diff-filter=ACMR "$since" -- "$@"
        git ls-files --others --exclude-standard -- "$@"
    else
        git ls-files -- "$@"
    fi | grep -vE '^third_party/|\.(patch|diff)$' | while IFS= read -r file; do
        [[ -L $file ]] || echo "$file"
    done
}

format() { # <tool> <pinned version> <pathspec...> -- <command...>: runs the command over the matching files
    local tool=$1 pinned=$2
    shift 2
    local patterns=()
    while [[ $1 != -- ]]; do
        patterns+=("$1")
        shift
    done
    shift
    if ! command -v "$tool" >/dev/null; then
        echo "  skipping $tool: not installed (./formatFiles.sh --install)"
        return
    fi
    local found
    found=$(versions | grep "^$tool " | cut -d' ' -f2)
    [[ $found == "$pinned" ]] || echo "  $tool $found is not the pinned $pinned: its output may differ from CI's" >&2
    local files
    mapfile -t files < <(collect "${patterns[@]}")
    if ((${#files[@]} == 0)); then
        echo "  $tool: nothing to do"
        return
    fi
    "$@" "${files[@]}" || true
    echo "  $tool: ${#files[@]} file(s)"
}

shellcheckFixes() { # applies the fixes shellcheck can make itself; findings without a fix are left to review
    shellcheck --format=diff "$@" 2>/dev/null | git apply --allow-empty 2>/dev/null || true
}

whitespace() { # no trailing blanks, exactly one final newline; a file already so is not touched, so nothing rebuilds
    local file
    for file in "$@"; do
        [[ -s $file ]] || continue
        if grep -q '[[:blank:]]$' "$file" || [[ -n $(tail -c1 "$file") || -z $(tail -c2 "$file" | tr -d '\n') ]]; then
            sed -i -e 's/[[:blank:]]*$//' -e :a -e '/^\n*$/{$d;N;ba' -e '}' "$file"
            [[ -z $(tail -c1 "$file") ]] || echo >>"$file"
        fi
    done
}

echo "formatting $([[ $changedOnly == true ]] && echo "the files changed since $since" || echo "all tracked files")"
format clang-format "$kClangFormat" '*.c' '*.cc' '*.cpp' '*.cxx' '*.h' '*.hh' '*.hpp' '*.hxx' -- clang-format -i
format cmake-format "$kCmakeFormat" '*CMakeLists.txt' '*.cmake' -- cmake-format -i
format prettier "$kPrettier" '*.md' '*.yml' '*.yaml' -- prettier --log-level warn --write
format shellcheck "$kShellcheck" '*.sh' -- shellcheckFixes
format black "$kBlack" '*.py' -- black --quiet
mapfile -t textFiles < <(collect . | xargs -r -d '\n' grep -Il '' --)
whitespace "${textFiles[@]}"
echo "  whitespace: ${#textFiles[@]} file(s)"
