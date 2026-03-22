# GNU Radio 4 - Installation Guide

## Introduction

GNU Radio 4 (GR4) is a modern C++ signal processing framework developed under the FAIR-ACC GNU Radio initiative.

Building the project is required to run tests, examples, and develop block libraries.

 

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

 

## Platform Notes

- Linux: Expected to work with a modern toolchain  
- Windows: See Windows setup instructions:  
  https://github.com/fair-acc/gnuradio4/pull/708  
- macOS: Currently not reliably supported  

 

## Troubleshooting

- Ensure CMake ≥ 3.28  
- Ensure compiler supports C++23  
- Check logs:  
  - CMakeFiles/CMakeError.log  
  - CMakeFiles/CMakeOutput.log  

For a reproducible setup, see Docker workflow:  
https://github.com/fair-acc/gnuradio4/blob/main/DEVELOPMENT.md#docker-cli