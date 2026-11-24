// wasm2wgsl: translates the WebGPU kernels of a linked Emscripten binary to WGSL compute pipelines.
//
// A kernel is every instantiation of `gr::device::webGpuKernel<TKernel>(const void*, uint32_t)`, found by name.
// Its call graph is emitted as WGSL functions over one storage buffer that mirrors wasm linear memory at the same
// addresses, so pointers stored inside a block's mirror need no translation. A construct WGSL cannot express (f64,
// SIMD, sub-word and wait/notify atomics, indirect calls, imports, recursion) refuses that kernel by name.
// usage: wasm2wgsl <in.wasm> <out.json> [--tint <tint>]
//   --tint: validate every translated kernel; one Tint rejects is refused with Tint's message, so a translator bug
//           becomes a named CPU fallback at run time rather than a failed pipeline

#include <array>
#include <cstdio>
#include <format>
#include <fstream>
#include <optional>
#include <print>
#include <ranges>
#include <string>
#include <string_view>
#include <vector>

#include "WgslKernel.hpp"

namespace {

using namespace wasm2wgsl;

std::string jsonString(std::string_view s) {
    std::string out = "\"";
    for (const char c : s) {
        switch (c) {
        case '"': out += "\\\""; break;
        case '\\': out += "\\\\"; break;
        case '\n': out += "\\n"; break;
        case '\t': out += "\\t"; break;
        default:
            if (static_cast<unsigned char>(c) < 0x20) {
                out += std::format("\\u{:04x}", static_cast<unsigned>(c));
            } else {
                out += c;
            }
        }
    }
    return out + "\"";
}
/// Tint's first error for `wgsl`, or nothing when it accepts the module
std::optional<std::string> tintRejects(const std::string& tint, const std::string& scratchFile, const std::string& wgsl) {
    std::ofstream(scratchFile) << wgsl;
    FILE* const pipe = popen(std::format("\"{}\" --format wgsl -o /dev/null \"{}\" 2>&1", tint, scratchFile).c_str(), "r");
    if (pipe == nullptr) {
        return std::format("cannot run Tint '{}'", tint);
    }
    std::string output;
    for (std::array<char, 512> chunk{}; std::fgets(chunk.data(), static_cast<int>(chunk.size()), pipe) != nullptr;) {
        output += chunk.data();
    }
    if (pclose(pipe) == 0) {
        return std::nullopt;
    }
    const std::size_t error = output.find("error");
    const std::size_t from  = error == std::string::npos ? 0UZ : error;
    return std::format("Tint rejects the generated WGSL: {}", output.substr(from, output.find('\n', from) - from));
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::println(stderr, "usage: {} <in.wasm> <out.json> [--tint <tint>]", argv[0]);
        return 2;
    }
    std::string tint;
    for (int a = 3; a < argc; ++a) {
        const std::string_view option = argv[a];
        if (option == "--tint" && a + 1 < argc) {
            tint = argv[++a];
        } else {
            std::println(stderr, "{}: unknown argument '{}'", argv[0], option);
            return 2;
        }
    }
    std::ifstream                   in(argv[1], std::ios::binary);
    const std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    Module                          module;
    try {
        module = parse(bytes);
    } catch (const Refusal& refusal) {
        std::println(stderr, "{}: {}", argv[1], refusal.reason);
        return 1;
    }

    std::vector<Kernel> kernels;
    for (const auto& [tableIndex, function] : module.tableToFunction) {
        if (function < module.functions.size() && module.functions[function].name.find("webGpuKernel") != std::string::npos) {
            kernels.push_back(translateKernel(module, tableIndex, function));
            if (Kernel& kernel = kernels.back(); !tint.empty() && kernel.refusal.empty()) {
                if (std::optional<std::string> rejected = tintRejects(tint, std::string(argv[2]) + ".check.wgsl", kernel.wgsl)) {
                    kernel.refusal = std::move(*rejected);
                    kernel.wgsl.clear();
                }
            }
        }
    }

    std::string json = "{\n  \"kernels\": [\n";
    for (std::size_t k = 0; k < kernels.size(); ++k) {
        const Kernel& kernel = kernels[k];
        json += std::format("    {{\"table\": {}, \"name\": {}, \"refused\": {}, \"stackBytes\": {}, \"inlinedInstructions\": {}, \"wgsl\": {}}}{}\n", kernel.tableIndex, jsonString(kernel.name), kernel.refusal.empty() ? "null" : jsonString(kernel.refusal), kernel.stackBytes, kernel.inlinedInstructions, jsonString(kernel.wgsl), k + 1 == kernels.size() ? "" : ",");
        std::println("{} table {:>5}: {}{}", kernel.refusal.empty() ? "translated" : "refused   ", kernel.tableIndex, kernel.refusal.empty() ? "" : kernel.refusal + " -- ", kernel.name.substr(0, 160));
    }
    json += "  ]\n}\n";
    if (std::ofstream out(argv[2]); !(out << json)) {
        std::println(stderr, "{}: cannot write", argv[2]);
        return 1;
    }
    const auto refused = std::ranges::count_if(kernels, [](const Kernel& k) { return !k.refusal.empty(); });
    std::println("{} kernels: {} translated, {} refused", kernels.size(), kernels.size() - static_cast<std::size_t>(refused), refused);
    return 0;
}
