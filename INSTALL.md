# GNU Radio 4 - Installation Guide

## Introduction

GNU Radio 4 (GR4) is a modern C++ signal processing framework developed under the FAIR-ACC GNU Radio initiative.

Building the project is required to run tests, examples, and to develop or use block libraries.



## Requirements

The following tools are required to build and use GNU Radio 4:

* **CMake ≥ 3.28**
* **C++20 compatible compiler**

  * GCC 11 or newer (Linux)
  * Clang 14 or newer (Linux, macOS)
  * MSVC 2022 (Windows)
* Git
* Python 3 (required for tests and tooling)

Verify tool versions:

```bash
cmake --version
g++ --version
```



## Obtaining the Source Code

Clone the official GNU Radio 4 repository:

```bash
git clone https://github.com/fair-acc/gnuradio4.git
cd gnuradio4
```



## Building GNU Radio 4

GNU Radio 4 uses an out-of-source CMake build.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build ./build
```



## Running Tests

After a successful build, tests can be executed using:

```bash
ctest --output-on-failure
```



## Using GNU Radio 4 via CMake FetchContent

GNU Radio 4 can be integrated into external CMake projects using `FetchContent`.

```cmake
include(FetchContent)

FetchContent_Declare(
  gnuradio4
  GIT_REPOSITORY https://github.com/fair-acc/gnuradio4.git
  GIT_TAG main
)

FetchContent_MakeAvailable(gnuradio4)

target_link_libraries(your_target PRIVATE gnuradio4::core)
```

Pinning a specific commit or tag is recommended for reproducible builds.



## Platform-Specific Notes

### Ubuntu 24.04

The default Ubuntu CMake package is insufficient. GNU Radio 4 requires **CMake ≥ 3.28**.

Install CMake from Kitware:

```bash
sudo apt update
sudo apt install -y ca-certificates gpg wget

wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc \
  | gpg --dearmor - \
  | sudo tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null

echo "deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] \
https://apt.kitware.com/ubuntu/ noble main" \
| sudo tee /etc/apt/sources.list.d/kitware.list

sudo apt update
sudo apt install -y cmake g++ git python3
```

Verify:

```bash
cmake --version
```
### Windows 10 / 11

Requirements:

* Visual Studio 2022 with C++ workload
* CMake ≥ 3.28 (added to PATH)
* Git for Windows
* Python 3

Configure and build:

```powershell
cmake -S . -B build
cmake --build build --config Release
```

### macOS

Using Homebrew:

```bash
brew update
brew install cmake llvm git python3
```

Build:

```bash
mkdir build && cd build
cmake ..
cmake --build .
```



## Troubleshooting

### CMake version too old

Ensure `cmake --version` reports **3.28 or newer**.

### Compiler does not support C++20

Upgrade the compiler to a supported version listed in the requirements.

### Build failures

Inspect the following files for diagnostics:

* `CMakeFiles/CMakeError.log`
* `CMakeFiles/CMakeOutput.log`


## Contributing

For development workflows and contribution guidelines, refer to:

* `DEVELOPMENT.md`
* `CONTRIBUTING.md`
