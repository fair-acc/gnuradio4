# GNU Radio 4 Installation Guide

## Introduction
GNU Radio 4 (GR4) is a modern, header-only C++ library for signal processing and software-defined radio applications. As a header-only library, **GR4 does not require traditional installation** - you can use it directly in your projects without complex build procedures.

## Usage Options

You have two main options for using GNU Radio 4:

### Option 1: Direct Usage
You can fork the repository and work directly within the library. This approach is especially useful if you:
- Plan to contribute to the core or block library
- Want to submit pull requests
- Need to modify the library for your specific needs

### Option 2: Including in Your Project (Recommended)
If you prefer to use the library "as-is" without modifying its code, you can include it in your projects via CMake using the `FetchContent_Declare(...)` statement:

```cmake
include(FetchContent)
FetchContent_Declare(
    gnuradio4
    GIT_REPOSITORY https://github.com/fair-acc/gnuradio4.git
    GIT_TAG main  # Consider using a specific tag or commit hash for stability
)
FetchContent_MakeAvailable(gnuradio4)

# Link your target with GR4
target_link_libraries(your_target PRIVATE gnuradio4::core)
```

**Note**: Installation is typically not required unless you need runtime polymorphism (i.e., a pre-built block library).

## System Requirements

### Prerequisites
Before beginning, please ensure your system meets the following requirements:
- **CMake** (version 3.28 or later)
- **C++ Compiler** supporting C++17 or later:
  - GCC 9+ (Linux)
  - Clang 10+ (Linux/macOS)
  - MSVC 2019+ (Windows)
- **Git** for version control
- **Python 3** (for testing and examples)

### Platform-Specific Setup

#### Ubuntu 24.04 or Later
For Ubuntu-based systems, install the required dependencies with:
```bash
sudo apt update
sudo apt install g++ cmake git python3
```

#### Windows 10/11
Install:
- Visual Studio 2019 or later with C++ development tools
- CMake (ensure it's added to PATH)
- Git for Windows
- Python 3

#### macOS
Using Homebrew:
```bash
brew update
brew install cmake llvm git python3
```

## Building Tests (Optional)
Since GR4 is header-only, building is only required for running tests:

```bash
git clone https://github.com/fair-acc/gnuradio4.git
cd gnuradio4
mkdir build && cd build
cmake ..
cmake --build . --config Release
```

## Verification and Testing
To verify your build works correctly:
```bash
cd build
ctest --output-on-failure
```

## Troubleshooting

### Common Issues

- **CMake Version**: Ensure you have CMake 3.28 or later
- **Compiler Support**: Verify your compiler supports C++17
- **Include Paths**: When using directly, make sure include paths are set correctly

If you encounter issues not covered here:
1. Check the repository's `DEVELOPMENT.md` file
2. Raise an issue on GitHub with detailed information

## Learning Resources

- **GR4 Workshop**: A GNU Radio Days workshop at FAIR (end of August) provides in-person and online (via YouTube) tutorials on GR4 and differences from GR3
- For Out-of-Tree (OOT) module development, refer to the [GR4 OOT Example Repository](https://github.com/fair-acc/gr4-examples)

## Additional Resources
- [GNU Radio Wiki](https://wiki.gnuradio.org)
- [GNU Radio Discourse Forum](https://discuss.gnuradio.org/)

---

This guide provides basic installation and usage instructions for GNU Radio 4. For specific questions or assistance with your project, please raise an issue on GitHub or join the GNU Radio community discussions.

