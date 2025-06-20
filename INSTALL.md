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

## Cloning the Repository
If you want to use or contribute to GR4 directly, clone the repository with:
```bash
git clone https://github.com/fair-acc/gnuradio4.git
cd gnuradio4
```

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

## Building GNU Radio 4 (Optional)
GR4 is header-only and does not require building for normal use. However, if you want to run tests or examples, you can build the project as follows:

```bash
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

## Contributing
We welcome contributions to GNU Radio 4! To contribute:
- Fork the repository
- Create a feature branch for your changes
- Commit and push your changes
- Open a pull request with a clear description

Please follow the project's coding standards and check the `CONTRIBUTING.md` file for more details.

## Learning Resources

- **GR4 Workshop**: A GNU Radio Days workshop at FAIR (end of August) provides in-person and online (via YouTube) tutorials on GR4 and differences from GR3
- For Out-of-Tree (OOT) module development, refer to the [GR4 OOT Example Repository](https://github.com/fair-acc/gr4-examples)

## Additional Resources
- [GNU Radio Wiki](https://wiki.gnuradio.org)
- [GNU Radio Discourse Forum](https://discuss.gnuradio.org/)

## For More Details
For more details, refer to the [GNU Radio Wiki](https://wiki.gnuradio.org) or join community discussions. You can also find more advanced topics in the `DEVELOPMENT.md` file in this repository.

## Community Getting Started Example

For a practical, community-driven guide to getting up and running with GNU Radio 4, we recommend exploring [@daniestevez's gr4-packet-modem repository](https://github.com/daniestevez/gr4-packet-modem). This repository provides a real-world example of how to set up and use GR4 in a digital communications application.

Additionally, you can watch the [GR4 Packet Modem Talk on YouTube](https://youtu.be/1EPuhaIjxCk?si=fWJdhCjhZchPS3O0), where the author explains the design, implementation, and lessons learned from building the first out-of-tree (OOT) module for GNU Radio 4. The talk covers:

- Modular implementation with many small blocks for reusability and testing
- Suitability for IP communication and low-latency operation
- Flexible packet sizes and support for different modulations
- Use of good signal processing techniques (e.g., FFT-based synchronization, LDPC coding for headers)
- Real-world integration examples, including SDR hardware and software loopback
- Practical feedback and insights on using and extending GR4 for new projects

These resources are a solid starting point for new users looking for step-by-step, practical guidance and deeper technical understanding.

**References:**
- [gr4-packet-modem repository](https://github.com/daniestevez/gr4-packet-modem)
- [GR4 Packet Modem Talk (YouTube)](https://youtu.be/1EPuhaIjxCk?si=fWJdhCjhZchPS3O0)

---

This guide provides basic installation and usage instructions for GNU Radio 4. For specific questions or assistance with your project, please raise an issue on GitHub or join the GNU Radio community discussions.

## Getting Up and Running Example

To help you get started with GNU Radio 4 in a real-world application, here is a practical example inspired by community experience:

### Example: Packet-Based QPSK Modem in GR4

This example demonstrates how to implement a modular, packet-based QPSK modem using GNU Radio 4. The design uses many small, reusable blocks, making it easy to adapt for your own projects.

#### Block Diagram (Transmitter & Receiver)

Transmitter:
```
[Packet Ingress]
      |
[Header Formatter]
      |
[Scrambler]
      |
[Modulator]
      |
[Burst/Stream Mode Switch]
      |
[Root Raised Cosine Filter]
      |
[SDR Output]
```

Receiver:
```
[SDR Input]
      |
[FFT Correlation]
      |
[Matched Filtering]
      |
[Costas Loop]
      |
[Header Parsing]
      |
[Payload Processing]
      |
[Packet Output]
```

#### Key Design Principles

- **Modular Blocks:** Each function (e.g., header formatting, scrambling, modulation, synchronization) is implemented as a separate block for flexibility and testing.
- **IP Communication:** The modem is suitable for sending IP packets, allowing you to use standard tools like `ping` or `iperf` to test your setup.
- **Low Latency:** The design supports both burst and stream transmission modes, with attention to minimizing latency.
- **Flexible Packet Size:** The modem supports variable packet sizes and can be extended with your own FEC (Forward Error Correction) if needed.
- **Robust Synchronization:** Uses FFT-based detection and synchronization for reliable packet detection, even at low SNR.

#### Example Flow (Transmitter)

1. **Packet Ingress:** Accepts packets (e.g., from a network interface).
2. **Header Formatter:** Prepares a header with packet length and type.
3. **Scrambler:** Applies scrambling to the payload for better signal properties.
4. **Modulator:** Maps bits to QPSK symbols.
5. **Burst/Stream Mode:** Supports both single-packet (burst) and continuous (stream) transmission.
6. **Root Raised Cosine Filter:** Shapes the signal for transmission.

#### Example Flow (Receiver)

1. **FFT Correlation:** Detects the packet preamble and estimates timing, frequency, and phase.
2. **Matched Filtering:** Improves SNR and prepares for symbol recovery.
3. **Costas Loop:** Tracks and corrects phase errors.
4. **Header Parsing:** Decodes the header to determine packet length and type.
5. **Payload Processing:** Applies descrambling, FEC (if present), and CRC checking.
6. **Packet Output:** Delivers recovered packets to the application.

#### Practical Tips

- **Testing:** You can use SDR hardware (e.g., USRP, RTL-SDR) or run software loopback simulations.
- **Latency Control:** Limit the number of packets in the flow graph to avoid excessive buffering and latency.
- **Customization:** The modular design allows you to add new modulations, FEC schemes, or adapt the modem for different applications.

#### Troubleshooting Tips (Memory/Buffer Issues)

- If you notice increasing RAM usage or high latency, check the number of packets buffered in the flow graph. Large buffers can cause excessive memory consumption and delay.
- Use smaller buffer sizes or implement backpressure mechanisms to prevent overfilling.
- Monitor for buffer state corruption, especially when using multi-threaded schedulers. If you encounter crashes, try switching to a single-threaded scheduler.
- Be aware that default buffer sizes (e.g., 65k elements) may not be optimal for all applications. Adjust as needed for your use case.

#### Example Use Case

- Send IP packets through the modem and verify end-to-end connectivity using `ping` or `iperf`.
- Observe and tune latency, SNR, and packet error rates by adjusting modem parameters and channel conditions.

#### Advanced Use Cases

- **Satellite Experiments:** The modular GR4 modem design has been successfully tested in satellite communication scenarios, including real-world uplink/downlink with SDR hardware and challenging SNR conditions.
- **Custom Schedulers:** For advanced users, experimenting with custom schedulers or flow graph-level latency control can further optimize performance for demanding applications.
- **Extensibility:** The design allows for easy integration of new modulations, FEC schemes, and application-specific features, making it suitable for both research and production environments.

This section provides a hands-on, practical starting point for new users, based on proven community experience with GNU Radio 4.

