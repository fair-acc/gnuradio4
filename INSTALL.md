# GNU Radio 4 Installation Guide

## Introduction
GNU Radio 4 (GR4) is a header-only library, which means it does not require traditional installation. Instead, you can use it directly in your projects. This guide provides instructions on how to set up and use GR4.

## Prerequisites
Ensure you have the following dependencies installed:
- CMake (version 3.16 or later)
- A C++ compiler supporting C++17 or later (GCC, Clang, or MSVC)
- Git
- Python (for optional testing)

## Cloning the Repository
To get started, clone the GNU Radio 4 repository:
```bash
 git clone https://github.com/fair-acc/gnuradio4.git
 cd gnuradio4
```

## Building GNU Radio 4
Since GR4 is a header-only library, you generally don’t need to build or install it. However, if you want to run the tests, follow these steps:
```bash
 mkdir build && cd build
 cmake ..
 make -j$(nproc)  # or 'cmake --build .' for multi-platform support
```

## Using GNU Radio 4 in Your Project
You can include GNU Radio 4 in your CMake-based project using `FetchContent`:
```cmake
include(FetchContent)
FetchContent_Declare(
    gnuradio4
    GIT_REPOSITORY https://github.com/fair-acc/gnuradio4.git
    GIT_TAG main  # Change to a specific tag if needed
)
FetchContent_MakeAvailable(gnuradio4)
```

## Running Tests
To verify the build, run:
```bash
 ctest --output-on-failure
```

## Contributing
If you want to contribute, fork the repository and submit a pull request. Follow the coding guidelines provided in the repository.

## Additional Resources
For more details, refer to the [GNU Radio Wiki](https://wiki.gnuradio.org) or join community discussions.

---

This guide is a basic starting point. If you encounter issues, please check the repository’s `DEVELOPMENT.md` or raise an issue on GitHub.

