# GNU Radio 4 - Installation Guide

At the moment GNU Radio 4 is not yet packaged and has to be installed from source.
This document describes the necessary steps and prerequisites to build, test and install gr4 to allow usage and block-library development.

As the installation and packaging matures, these instructions are expected to be extended.

## Requirements

- CMake ≥ 3.28
- C++23 compatible compiler
  - GCC ≥ 14.2 (Linux)
- Git
- Python 3 (optional)

Verify tool versions:

cmake --version
g++ --version

## Clone Repository

git clone https://github.com/fair-acc/gnuradio4.git
cd gnuradio4

## Build

GNU Radio 4 uses an out-of-source CMake build.

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build

## Run Tests

cd build
ctest --output-on-failure

## Ubuntu 24.04 (Dependencies)

sudo apt update
sudo apt install -y cmake g++ git python3 python3-dev

Then follow the build steps above.

## MacOS via Homebrew

MacOS builds are supported by using `llvm@20` from the [Homebrew package manager](https://brew.sh/):

## Platform Notes

- Linux: Expected to work with a modern toolchain
- Windows: See Windows setup instructions:
  https://github.com/fair-acc/gnuradio4/blob/main/DEVELOPMENT.md#win32-development-environment---msys2
- macOS: supported using llvm@20 from homebrew

## Troubleshooting

- Ensure CMake ≥ 3.28
- Ensure compiler supports C++23
- Check logs:
  - CMakeFiles/CMakeError.log
  - CMakeFiles/CMakeOutput.log

For a reproducible setup, see Docker workflow:
https://github.com/fair-acc/gnuradio4/blob/main/DEVELOPMENT.md#docker-cli
