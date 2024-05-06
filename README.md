<p align="center">
<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-1-orange.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->
<img src="docs/logo.png" width="65%" />
</p>

[![License](https://img.shields.io/badge/License-LGPL%203.0-blue.svg)](https://opensource.org/licenses/LGPL-3.0)
![CMake](https://github.com/fair-acc/graph-prototype/workflows/CMake/badge.svg)

# GNU Radio 4.0 prototype

> [!IMPORTANT]
> This is the GNU Radio 4.0 (GR4) prototype and is currently in a beta state. For production use, 
> please use the GNU Radio 3.X (GR3) version found [here](https://github.com/gnuradio/gnuradio).
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
  * Specifically for discussions related to GNURadio 4.0 join the [#architecture channel](https://matrix.to/#/#architecture:gnuradio.org)
* [Contributors and Affiliated Organizations](https://github.com/gnuradio/gnuradio/blob/main/CONTRIBUTORS.md)

## What's New in GNU Radio 4.0?

- **Smooth Transition with Updated GR3 Paradigms**: User-defined blocks and flow-graphs continue to be accessible graphically, through Python, and using C++. Notable simplifications and streamlining have been made to the underlying low-level architecture and design, as described [here](https://github.com/fair-acc/graph-prototype/tree/main/core).
- **Enhanced Data Types**: Support for fundamental data types (i.e. ints, floats, complex numbers) as well as more complex, structured, and user-defined custom types.
- **Simplified Block Development**: Modern C++ and computing standards enable rapid development.
- **High-Performance Signal-Processing**: Significant performance boosts from lock-free buffers, compile-time optimisations, and built-in SIMD & SYCL support.
- **Flexible Scheduling**: An updated scheduling method improves performance and enables user-defined schedulers to balance throughput, parallelism, and latency depending on the application.
- **Recursive Directed Graphs**: Support for basic feedback loops.
- **Broadened Hardware Support**: Ready for CPUs, MCUs, GPUs, and aspirationally FPGAs.
- **Bridging R&D and Industrial Use**: Aims to bridge the gap between academics using GNU Radio for research, hobbyists using it for prototyping and safe operational use by research organizations and industry.

## License and Copyright

Unless otherwise noted: SPDX-License-Identifier: LGPL-3.0-or-later
All code contributions to GNU Radio will be integrated into a library under the LGPL, ensuring it remains free/libre (FLOSS) for both personal and commercial use, without further constraints on either.
For details on how to contribute, please consult: [CONTRIBUTING.md](CONTRIBUTING.md)

Copyright (C) 2001-September 2020 GNU Radio Project -- managed by Free Software Foundation, Inc.  
Copyright (C) September 2020-2024 GNU Radio Project -- managed by SETI Institute  
Copyright (C) 2018-2024 FAIR -- Facility for Antiproton & Ion Research, Darmstadt, Germany


## Acknowledgements

The GNU Radio project appreciates the contributions from FAIR in the co-development of GNU Radio 4.0. Their dedicated efforts have played a key role in enhancing the capabilities of our open-source SDR technology. 
We would like to recognize the following contributors for their roles in redesigning the core that has evolved into GR 4.0:

 * Ivan Čukić <ivan.cukic@kdab.com>
 * Matthias Kretz <M.Kretz@GSI.de>
 * Alexander Krimm, <A.Krimm@GSI.de> 
 * Semen Lebedev, <S.Lebedev@GSI.de>
 * Frank Osterfeld, <Frank.Osterfeld@kdab.com>
 * Ralph J. Steinhagen, <R.Steinhagen@GSI.de>
## Contributors ✨

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/frankosterfeld"><img src="https://avatars.githubusercontent.com/u/483854?v=4?s=100" width="100px;" alt="Frank Osterfeld"/><br /><sub><b>Frank Osterfeld</b></sub></a><br /><a href="https://github.com/fair-acc/gnuradio4/commits?author=frankosterfeld" title="Code">💻</a> <a href="#design-frankosterfeld" title="Design">🎨</a> <a href="https://github.com/fair-acc/gnuradio4/pulls?q=is%3Apr+reviewed-by%3Afrankosterfeld" title="Reviewed Pull Requests">👀</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!