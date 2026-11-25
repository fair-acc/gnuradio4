#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

#include <boost/ut.hpp>

#include "WgslKernel.hpp"

namespace {

using Bytes = std::vector<std::uint8_t>;

void appendUleb(Bytes& out, std::uint64_t value) {
    do {
        const auto low = static_cast<std::uint8_t>(value & 0x7FU);
        value >>= 7U;
        out.push_back(value == 0U ? low : static_cast<std::uint8_t>(low | 0x80U));
    } while (value != 0U);
}

void appendName(Bytes& out, std::string_view name) {
    appendUleb(out, name.size());
    out.insert(out.end(), name.begin(), name.end());
}

void appendSection(Bytes& module, std::uint8_t id, const Bytes& content) {
    module.push_back(id);
    appendUleb(module, content.size());
    module.insert(module.end(), content.begin(), content.end());
}

/// a module of one kernel `webGpuKernel` (i32 closure, i32 index) -> () at table slot 1 with `body` (its code after the
/// locals, `end` included), and `__stack_pointer` as global 0; `elementOffset` is the element segment's constant
/// expression, an `i32.const 1` unless a test asks otherwise
Bytes kernelModule(const Bytes& body, const Bytes& elementOffset = {0x41, 0x01, 0x0B}) {
    Bytes module{0x00, 0x61, 0x73, 0x6D, 0x01, 0x00, 0x00, 0x00};
    appendSection(module, 1, {0x01, 0x60, 0x02, 0x7F, 0x7F, 0x00});             // one type: (i32, i32) -> ()
    appendSection(module, 3, {0x01, 0x00});                                     // one function of that type
    appendSection(module, 4, {0x01, 0x70, 0x00, 0x02});                         // a funcref table of two slots
    appendSection(module, 6, {0x01, 0x7F, 0x01, 0x41, 0x80, 0x80, 0x04, 0x0B}); // mutable i32 = 65536
    Bytes elements{0x01, 0x00};                                                 // one active segment, table 0
    elements.insert(elements.end(), elementOffset.begin(), elementOffset.end());
    elements.insert(elements.end(), {0x01, 0x00}); // holding function 0
    appendSection(module, 9, elements);
    Bytes function{0x00}; // no locals
    function.insert(function.end(), body.begin(), body.end());
    Bytes code{0x01};
    appendUleb(code, function.size());
    code.insert(code.end(), function.begin(), function.end());
    appendSection(module, 10, code);
    Bytes names;
    appendName(names, "name");
    Bytes functionNames{0x01, 0x00};
    appendName(functionNames, "webGpuKernel");
    names.push_back(0x01);
    appendUleb(names, functionNames.size());
    names.insert(names.end(), functionNames.begin(), functionNames.end());
    Bytes globalNames{0x01, 0x00};
    appendName(globalNames, "__stack_pointer");
    names.push_back(0x07);
    appendUleb(names, globalNames.size());
    names.insert(names.end(), globalNames.begin(), globalNames.end());
    appendSection(module, 0, names);
    return module;
}

wasm2wgsl::Kernel translated(const Bytes& module) {
    const wasm2wgsl::Module parsed = wasm2wgsl::parse(module);
    return wasm2wgsl::translateKernel(parsed, 1U, parsed.tableToFunction.at(1U));
}

std::string refusalOf(auto&& action) {
    try {
        action();
    } catch (const wasm2wgsl::Refusal& refusal) {
        return refusal.reason;
    }
    return {};
}

} // namespace

const boost::ut::suite<"WgslKernel"> wgslKernelTests = [] {
    using namespace boost::ut;

    "a kernel that only returns translates, with no stack"_test = [] {
        const wasm2wgsl::Kernel kernel = translated(kernelModule({0x0B}));
        expect(kernel.refusal.empty()) << kernel.refusal;
        expect(eq(kernel.stackBytes, 0U));
        expect(kernel.wgsl.find("fn main") != std::string::npos);
    };

    "a frame of 48 bytes takes the deepest frame plus the 16-byte margin, 16-aligned: 64 bytes per work item"_test = [] {
        // global.get sp; i32.const 48; i32.sub; global.set sp; global.get sp; i32.const 48; i32.add; global.set sp; end
        const wasm2wgsl::Kernel kernel = translated(kernelModule({0x23, 0x00, 0x41, 0x30, 0x6B, 0x24, 0x00, 0x23, 0x00, 0x41, 0x30, 0x6A, 0x24, 0x00, 0x0B}));
        expect(kernel.refusal.empty()) << kernel.refusal;
        expect(eq(kernel.stackBytes, 64U));
    };

    "an f64 constant refuses the kernel, saying so"_test = [] {
        const wasm2wgsl::Kernel kernel = translated(kernelModule({0x44, 0, 0, 0, 0, 0, 0, 0, 0, 0x1A, 0x0B})); // f64.const 0; drop
        expect(kernel.refusal.find("f64") != std::string::npos) << kernel.refusal;
        expect(kernel.wgsl.empty());
    };

    "a kernel calling itself is refused for recursion, which WGSL forbids"_test = [] {
        const wasm2wgsl::Kernel kernel = translated(kernelModule({0x20, 0x00, 0x20, 0x01, 0x10, 0x00, 0x0B})); // call 0 (args)
        expect(kernel.refusal.find("recursion") != std::string::npos) << kernel.refusal;
    };

    "a truncated module is refused, not read past its end"_test = [] {
        const Bytes module = kernelModule({0x0B});
        expect(refusalOf([&] { std::ignore = wasm2wgsl::parse(std::span(module).first(module.size() - 3UZ)); }) == "truncated wasm");
    };

    "an element segment placed at instantiation (a relocatable module) refuses the binary"_test = [] {
        const Bytes module = kernelModule({0x0B}, {0x23, 0x00, 0x0B}); // offset: global.get 0
        expect(refusalOf([&] { std::ignore = wasm2wgsl::parse(module); }).find("relocatable") != std::string::npos);
    };
};

int main() { return boost::ut::cfg<>.run({.report_errors = true}) ? 1 : 0; }
