[![License](https://img.shields.io/badge/License-LGPL%203.0-blue.svg)](https://opensource.org/licenses/LGPL-3.0)
![CMake](https://github.com/fair-acc/graph-prototype/workflows/CMake/badge.svg)
# GNURadio 4.0 prototype
A small proof-of-concept for evaluating efficient [directed graph](https://en.wikipedia.org/wiki/Directed_graph)-based algorithms, notably required block 
('nodes' in graph-theory) structures, scheduling interfaces, and partial compile-time merging of
[directed acyclic](https://en.wikipedia.org/wiki/Directed_acyclic_graph) as well as 
[cyclic graphs](https://en.wikipedia.org/wiki/Feedback_arc_set) (aka. feedback loops).  

The expressed goal is to guide the low-level API design and functionality for the upcoming
[GNU Radio 4.0](https://github.com/gnuradio/gnuradio/tree/dev-4.0) release.

A [single header version](https://raw.githubusercontent.com/fair-acc/graph-prototype/single-header/singleheader/Graph.hpp)
is provided in the `singleheader/`subdirectory of the single-header branch.
If you want to locally regenerate the single header file, just follow
the [CI step](https://github.com/fair-acc/graph-prototype/blob/main/.github/workflows/single-header.yml#L38-L41).
It can be used on [compiler-explorer](https://compiler-explorer.com/z/EG7Eb9K83) with `-sdt=c++23 -O3` compiler options.

## Copyright & License
Copyright (C) 2018-2022 FAIR -- Facility for Antiproton & Ion Research, Darmstadt, Germany<br/>
Unless otherwise noted: [SPDX-License-Identifier: LGPL-3.0-or-later](https://spdx.org/licenses/LGPL-3.0-or-later.html)

### Contributors
 * Ivan Čukić <ivan.cukic@kdab.com>
 * Matthias Kretz <M.Kretz@GSI.de>
 * Alexander Krimm, <A.Krimm@GSI.de> 
 * Ralph J. Steinhagen, <R.Steinhagen@GSI.de>

### Acknowledgements
...

