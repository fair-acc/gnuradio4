// the parts of a linked wasm binary the translator reads: types, functions with their bodies, globals, the function table
// and the name section; anything it cannot read refuses the whole binary
#ifndef WASM2WGSL_WASM_MODULE_HPP
#define WASM2WGSL_WASM_MODULE_HPP

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cxxabi.h>
#include <format>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace wasm2wgsl {

enum class ValType : std::uint8_t { i32 = 0x7F, i64 = 0x7E, f32 = 0x7D, f64 = 0x7C, v128 = 0x7B, funcref = 0x70, externref = 0x6F };

struct Refusal {
    std::string reason;
};

inline constexpr std::size_t kMaxLocals = 50'000UZ; // the WebAssembly JS API's implementation limit per function

struct Reader {
    std::span<const std::uint8_t> bytes;
    std::size_t                   pos = 0;

    [[nodiscard]] bool atEnd() const { return pos >= bytes.size(); }

    std::uint8_t byte() {
        if (pos >= bytes.size()) {
            throw Refusal{"truncated wasm"};
        }
        return bytes[pos++];
    }
    std::uint64_t uleb() {
        std::uint64_t result = 0;
        for (unsigned shift = 0;; shift += 7) {
            if (shift >= 64U) {
                throw Refusal{"malformed wasm: an over-long LEB128"};
            }
            const std::uint8_t b = byte();
            result |= static_cast<std::uint64_t>(b & 0x7FU) << shift;
            if ((b & 0x80U) == 0) {
                return result;
            }
        }
    }
    std::int64_t sleb() {
        std::int64_t result = 0;
        unsigned     shift  = 0;
        std::uint8_t b      = 0;
        do {
            if (shift >= 64U) {
                throw Refusal{"malformed wasm: an over-long LEB128"};
            }
            b = byte();
            result |= static_cast<std::int64_t>(b & 0x7FU) << shift;
            shift += 7;
        } while ((b & 0x80U) != 0);
        if (shift < 64 && (b & 0x40U) != 0) {
            result |= -(static_cast<std::int64_t>(1) << shift);
        }
        return result;
    }
    std::uint32_t u32() { return static_cast<std::uint32_t>(uleb()); }
    std::string   name() {
        const std::span<const std::uint8_t> s = take(u32());
        return {reinterpret_cast<const char*>(s.data()), s.size()};
    }
    std::span<const std::uint8_t> take(std::size_t n) {
        if (n > bytes.size() - pos) {
            throw Refusal{"truncated wasm"};
        }
        auto s = bytes.subspan(pos, n);
        pos += n;
        return s;
    }
    std::uint32_t fixed32() {
        std::uint32_t v = 0;
        for (unsigned i = 0; i < 4; ++i) {
            v |= static_cast<std::uint32_t>(byte()) << (8 * i);
        }
        return v;
    }
};

struct FuncType {
    std::vector<ValType> params;
    std::vector<ValType> results;
};

struct Function {
    std::uint32_t                 typeIndex = 0;
    std::string                   importName; // `module.field`, empty for a function the module defines
    std::vector<ValType>          locals;     // params excluded
    std::span<const std::uint8_t> body;
    std::string                   name;
};

struct Global {
    ValType      type      = ValType::i32;
    bool         isMutable = false;
    bool         imported  = false;
    std::int64_t init      = 0;
    std::string  name;
};

struct Module {
    std::vector<FuncType>                  types;
    std::vector<Function>                  functions;
    std::vector<Global>                    globals;
    std::map<std::uint32_t, std::uint32_t> tableToFunction;
    std::optional<std::uint32_t>           stackPointer;
};

/// nullopt for a `global.get`, whose value is set only at instantiation
inline std::optional<std::int64_t> constExpr(Reader& r) {
    std::optional<std::int64_t> value = 0;
    for (;;) {
        const std::uint8_t op = r.byte();
        switch (op) {
        case 0x41: value = r.sleb(); break;
        case 0x42: value = r.sleb(); break;
        case 0x43: value = r.fixed32(); break;
        case 0x44: r.take(8); break;
        case 0x23:
            r.u32();
            value.reset();
            break;
        case 0xD2: value = r.u32(); break;
        case 0xD0: r.byte(); break;
        case 0x0B: return value;
        default: throw Refusal{std::format("unsupported constant-expression opcode 0x{:02x}", op)};
        }
    }
}

inline Module parse(std::span<const std::uint8_t> bytes) {
    Reader r{bytes};
    if (r.fixed32() != 0x6D736100U || r.fixed32() != 1U) {
        throw Refusal{"not a wasm module"};
    }
    enum class SectionId : std::uint8_t { custom = 0, type = 1, import = 2, function = 3, global = 6, element = 9, code = 10 };
    enum class ImportKind : std::uint8_t { function = 0, table = 1, memory = 2, global = 3 };
    constexpr std::uint8_t kFunctionNames = 1U; // subsections of the custom "name" section
    constexpr std::uint8_t kGlobalNames   = 7U;
    Module                 m;
    while (!r.atEnd()) {
        const std::uint8_t  id   = r.byte();
        const std::uint32_t size = r.u32();
        Reader              s{r.take(size)};
        switch (static_cast<SectionId>(id)) {
        case SectionId::custom: {
            if (s.name() != "name") {
                break;
            }
            while (!s.atEnd()) {
                const std::uint8_t sub = s.byte();
                Reader             part{s.take(s.u32())};
                if (sub == kFunctionNames || sub == kGlobalNames) {
                    for (std::uint32_t n = part.u32(); n > 0; --n) {
                        const std::uint32_t index = part.u32();
                        std::string         label = part.name();
                        if (sub == kFunctionNames && index < m.functions.size()) {
                            m.functions[index].name = std::move(label);
                        } else if (sub == kGlobalNames && index < m.globals.size()) {
                            m.globals[index].name = std::move(label);
                        }
                    }
                }
            }
            break;
        }
        case SectionId::type:
            for (std::uint32_t n = s.u32(); n > 0; --n) {
                if (s.byte() != 0x60) {
                    throw Refusal{"unsupported type form"};
                }
                FuncType t;
                for (std::uint32_t k = s.u32(); k > 0; --k) {
                    t.params.push_back(static_cast<ValType>(s.byte()));
                }
                for (std::uint32_t k = s.u32(); k > 0; --k) {
                    t.results.push_back(static_cast<ValType>(s.byte()));
                }
                m.types.push_back(std::move(t));
            }
            break;
        case SectionId::import: {
            const auto skipLimits = [&s] {
                const std::uint8_t flags = s.byte();
                s.u32();
                if ((flags & 1U) != 0) {
                    s.u32(); // the maximum
                }
            };
            for (std::uint32_t n = s.u32(); n > 0; --n) {
                std::string      moduleName = s.name();
                std::string      field      = s.name();
                const ImportKind kind       = static_cast<ImportKind>(s.byte());
                if (kind == ImportKind::function) {
                    m.functions.push_back(Function{.typeIndex = s.u32(), .importName = moduleName + "." + field, .locals = {}, .body = {}, .name = field});
                } else if (kind == ImportKind::table) {
                    s.byte(); // the reference type
                    skipLimits();
                } else if (kind == ImportKind::memory) {
                    skipLimits();
                } else if (kind == ImportKind::global) {
                    const auto type = static_cast<ValType>(s.byte());
                    m.globals.push_back(Global{.type = type, .isMutable = s.byte() != 0, .imported = true, .init = 0, .name = field});
                } else { // a tag
                    s.byte();
                    s.u32();
                }
            }
            break;
        }
        case SectionId::function:
            for (std::uint32_t n = s.u32(); n > 0; --n) {
                m.functions.push_back(Function{.typeIndex = s.u32(), .importName = {}, .locals = {}, .body = {}, .name = {}});
            }
            break;
        case SectionId::global:
            for (std::uint32_t n = s.u32(); n > 0; --n) {
                const auto type      = static_cast<ValType>(s.byte());
                const bool isMutable = s.byte() != 0;
                m.globals.push_back(Global{.type = type, .isMutable = isMutable, .imported = false, .init = constExpr(s).value_or(0), .name = {}});
            }
            break;
        case SectionId::element:
            for (std::uint32_t n = s.u32(); n > 0; --n) {
                const std::uint32_t flags         = s.u32();
                const bool          active        = (flags & 1U) == 0;
                const bool          explicitTable = (flags & 2U) != 0;
                const bool          byExpression  = (flags & 4U) != 0;
                std::int64_t        offset        = 0;
                if (active) {
                    if (explicitTable) {
                        s.u32(); // table index
                    }
                    offset = constExpr(s).value_or(-1);
                    if (offset < 0) {
                        throw Refusal{"an element segment's offset is set at instantiation (a relocatable module), so its table slots are unknown"};
                    }
                }
                if (!active || explicitTable) {
                    s.byte(); // elemkind or reftype
                }
                for (std::uint32_t k = s.u32(), slot = 0; slot < k; ++slot) {
                    const std::uint32_t function = byExpression ? static_cast<std::uint32_t>(constExpr(s).value_or(-1)) : s.u32();
                    if (active) {
                        m.tableToFunction[static_cast<std::uint32_t>(offset) + slot] = function;
                    }
                }
            }
            break;
        case SectionId::code: {
            const std::size_t firstDefined = m.functions.size() - s.u32();
            for (std::size_t index = firstDefined; index < m.functions.size(); ++index) {
                Reader body{s.take(s.u32())};
                for (std::uint32_t groups = body.u32(); groups > 0; --groups) {
                    const std::uint32_t count = body.u32();
                    const auto          type  = static_cast<ValType>(body.byte());
                    if (count > kMaxLocals - m.functions[index].locals.size()) {
                        throw Refusal{std::format("malformed wasm: more than {} locals in one function", kMaxLocals)};
                    }
                    m.functions[index].locals.insert(m.functions[index].locals.end(), count, type);
                }
                m.functions[index].body = body.bytes.subspan(body.pos);
            }
            break;
        }
        default: break;
        }
    }
    if (const auto named = std::ranges::find(m.globals, std::string_view{"__stack_pointer"}, &Global::name); named != m.globals.end()) {
        m.stackPointer = static_cast<std::uint32_t>(std::distance(m.globals.begin(), named));
    }
    if (!m.stackPointer && !m.globals.empty() && m.globals[0].isMutable && m.globals[0].type == ValType::i32) {
        m.stackPointer = 0; // Emscripten's convention when the name section carries no global names
    }
    if (std::ranges::any_of(m.functions, [&m](const Function& function) { return function.typeIndex >= m.types.size(); })) {
        throw Refusal{"malformed wasm: a function type index out of range"};
    }
    return m;
}

inline std::string demangle(const std::string& name) {
    int                                    status = 0;
    std::unique_ptr<char, decltype(&free)> text(abi::__cxa_demangle(name.c_str(), nullptr, nullptr, &status), &free);
    return status == 0 && text ? std::string(text.get()) : name;
}

} // namespace wasm2wgsl

#endif // WASM2WGSL_WASM_MODULE_HPP
