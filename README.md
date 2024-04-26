<p align="center">
<img src="docs/logo.png" width="65%" />
</p>

[![License](https://img.shields.io/badge/License-LGPL%203.0-blue.svg)](https://opensource.org/licenses/LGPL-3.0)
![CMake](https://github.com/fair-acc/graph-prototype/workflows/CMake/badge.svg)

# GNU Radio 4.0 prototype

> [!IMPORTANT]
> This is the repository containing the prototype of GNU Radio 4.0.
> It is currently in a beta state and not ready for production use.
> The main organization and repository can be found [here](https://github.com/gnuradio/gnuradio)
> Bug reports related to this beta should be submitted [here](https://github.com/fair-acc/graph-prototype/issues), 
> and bug reports for GNU Radio 3.X should be submitted [here](https://github.com/gnuradio/gnuradio/issues)

GNU Radio is a free & open-source signal processing runtime and signal processing
software development toolkit. Originally developed for use with software-defined
radios and for simulating wireless communications, it's robust capabilities have
led to adoption in hobbyist, academic, and commercial environments. GNU Radio has
found use in software-defined radio, digital communications, nuclear physics, high-
energy particle physics, astrophysics, radio astronomy and more!

## Helpful Links

* [GNU Radio Website](https://gnuradio.org)
* [GNU Radio Wiki](https://wiki.gnuradio.org/)
* [Github issue tracker for bug reports and feature requests](https://github.com/fair-acc/graph-prototype/issues)
* [View the GNU Radio Mailing List Archive](https://lists.gnu.org/archive/html/discuss-gnuradio/)
* [Subscribe to the GNU Radio Mailing List](https://lists.gnu.org/mailman/listinfo/discuss-gnuradio)
* [GNU Radio Chatroom on Matrix](https://chat.gnuradio.org/)
* [Contributors and Affiliated Organizations](https://github.com/gnuradio/gnuradio/blob/main/CONTRIBUTORS.md)

## What's New in GNU Radio 4.0?

- **Seamless Transition by Maintaining GR3 Paradigms**: User-defined blocks and flow-graphs accessible graphically, via Python, and using C++.
- **Enhanced Data Types**: Support for fundamental data types (i.e. ints, floats, complex numbers) as well as more complex, structured, and user-defined custom types.
- **Simplified Block Development**: Modern C++ and computing standards enable rapid development.
- **High-Performance Signal-Processing**: Significant performance boosts from lock-free buffers, compile-time optimisations, and built-in SIMD & SYCL support.
- **Flexible Scheduling**: An updated scheduling method improves performance and enables user-defined schedulers to balance throughput, parallelism, and latency depending on the application.
- **Recursive Directed Graphs**: Support for basic feedback loops.
- **Broadened Hardware Support**: Ready for CPUs, MCUs, GPUs, and aspirationally FPGAs.
- **Bridging R&D and Industrial Use**: Aims to bridge the gap between academics using GNU Radio for research, hobbyists using it for prototyping and safe operational use by research organizations and industry.
- [**More on architecture and design**](https://github.com/fair-acc/graph-prototype/tree/main/core)

## Copyright & License
Copyright (C) 2018-2024 FAIR -- Facility for Antiproton & Ion Research, Darmstadt, Germany<br/>
Unless otherwise noted: [SPDX-License-Identifier: LGPL-3.0-or-later](https://spdx.org/licenses/LGPL-3.0-or-later.html)

### Contributors
 * [GNU Radio Project](https://gnuradio.org)
 * Ivan Čukić <ivan.cukic@kdab.com>
 * Matthias Kretz <M.Kretz@GSI.de>
 * Alexander Krimm, <A.Krimm@GSI.de> 
 * Ralph J. Steinhagen, <R.Steinhagen@GSI.de>

### Acknowledgements
...

