---
name: restyle
description: Check or apply the exact formatter set the Restyled CI bot runs (clang-format v18, cmake-format, prettier for markdown and yaml, shellcheck, whitespace, black). Use before committing, or when a Restyled check fails.
---

# Restyle

`.restyled.yaml` pins seven restylers. Restyled checks the **PR head only**, so a per-commit
difference is not a gate failure.

| file                       | tool              |
| -------------------------- | ----------------- |
| `*.cpp` `*.hpp`            | `clang-format-18` |
| `CMakeLists.txt` `*.cmake` | `cmake-format`    |
| `*.md` `*.yml`             | prettier          |
| `*.sh`                     | shellcheck        |
| `*.py`                     | `black`           |

## Two traps

**Config resolves from the file's path.** Checking a copy under `/tmp` reports every file as
mis-formatted, because the tool never finds `.clang-format` / `.cmake-format.yaml`. Always check in
place, or pass `--assume-filename`.

**`cmake-format` picks its dialect from the filename.** A copy named `CMakeLists.txt.bak` parses as
something else and reports spurious differences. Keep the basename.

## Working tree

```bash
clang-format-18 -i <files>
cmake-format -i <files>          # idempotent: a second run must be a no-op
~/venvs/ort-gpu/bin/python -m black <files>
```

## A specific commit

```bash
git show $c:$f | clang-format-18 --assume-filename="$f" --dry-run --Werror
git show $c:$f | python -m black --quiet --check -
```

For CMake files, write to a scratch directory _inside the repo_ keeping the basename, format there,
compare, delete.
