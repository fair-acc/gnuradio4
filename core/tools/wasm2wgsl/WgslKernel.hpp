// one kernel's call graph as one WGSL compute pipeline over the mirrored heap, or the reason it has none
#ifndef WASM2WGSL_WGSL_KERNEL_HPP
#define WASM2WGSL_WGSL_KERNEL_HPP

#include <algorithm>
#include <array>
#include <cstdint>
#include <format>
#include <map>
#include <optional>
#include <ranges>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "WasmModule.hpp"

namespace wasm2wgsl {

inline std::string_view wgslType(ValType t) {
    switch (t) {
    case ValType::i32: return "i32";
    case ValType::f32: return "f32";
    case ValType::i64: return "vec2<u32>"; // (low, high) words, the helpers `i64_*` in the prelude
    case ValType::f64: throw Refusal{"double precision (f64) has no WGSL type"};
    case ValType::v128: throw Refusal{"SIMD (v128) is not translated"};
    default: throw Refusal{"reference types are not translated"};
    }
}

inline std::string zeroOf(ValType t) { return t == ValType::f32 ? "0.0f" : t == ValType::i64 ? "vec2<u32>(0u, 0u)" : "0i"; }

// single-precision libm functions whose musl bodies compute in f64 (so would refuse the kernel): each becomes the
// WGSL built-in of the same meaning, or the translator's own where WGSL has none or a looser one
struct LibmBuiltin {
    std::string_view libm;
    std::string_view wgsl;
};
constexpr std::array<LibmBuiltin, 20> kLibmBuiltins{{
    {"sinf", "gr_sin"},                                                                                                                                                                                                                                                                                                                                                           // Cephes with an error-free reduction: within 2 ulp for |x| < 8192
    {"cosf", "gr_cos"}, {"tanf", "tan"}, {"asinf", "asin"}, {"acosf", "acos"}, {"atanf", "atan"}, {"atan2f", "atan2"}, {"sinhf", "sinh"}, {"coshf", "cosh"}, {"tanhf", "tanh"}, {"expf", "exp"}, {"exp2f", "exp2"}, {"logf", "log"}, {"log2f", "log2"}, {"sqrtf", "sqrt"}, {"fabsf", "abs"}, {"floorf", "floor"}, {"ceilf", "ceil"}, {"truncf", "trunc"}, {"hypotf", "gr_hypot"}, // without overflow, within 2 ulp measured on AMD and SwiftShader: WGSL bounds division and sqrt loosely
}};

/// Emscripten's exception runtime that only a landing pad calls. A landing pad runs only when `__THREW__` reads non-zero
/// after an `invoke_*`, and on the GPU every invoke is a direct call that cannot throw, so these calls are dead code
/// (LLVM's WebAssemblyLowerEmscriptenEHSjLj.cpp; `__clang_call_terminate` is clang's handler for a `noexcept` one, and
/// `_emscripten_tempret_get` returns the catch selector, its only use when i64 crosses to JavaScript as BigInt)
[[nodiscard]] inline bool isLandingPadRuntime(std::string_view name) {
    constexpr std::array<std::string_view, 6> kExact{"__cxa_begin_catch", "__cxa_end_catch", "__cxa_rethrow", "__resumeException", "__clang_call_terminate", "_emscripten_tempret_get"};
    return name.starts_with("__cxa_find_matching_catch_") || std::ranges::contains(kExact, name);
}

inline std::string i32Literal(std::int64_t v) {
    const auto bits = static_cast<std::uint32_t>(v);
    return static_cast<std::int32_t>(bits) >= 0 ? std::format("{}i", static_cast<std::int32_t>(bits)) : std::format("bitcast<i32>(0x{:08x}u)", bits);
}

/// translates one wasm function body to a WGSL function; counts its call sites per direct callee and its translated instructions
struct FunctionEmitter {
    const Module&                           module;
    std::uint32_t                           index;
    std::map<std::uint32_t, std::uint32_t>& callees;      // callee -> call sites in this function
    std::uint32_t&                          instructions; // reachable wasm instructions, those translated
    bool&                                   writesLanes;  // stores a byte or half-word somewhere: the kernel needs the atomic heap
    bool&                                   readsThrew;   // reads Emscripten's `__THREW__`: each work item needs its own word for it
    std::uint32_t&                          frameBytes;   // bytes this function moves the stack pointer down by, the deepest `global.set` of it

    struct Value {
        std::string                 expr;
        ValType                     type;
        std::optional<std::int64_t> constant{};     // an `i32.const`: an `invoke_*` names its target by one
        std::optional<std::int64_t> belowSp{};      // derived from the stack pointer by constants: bytes below its value at entry
        bool                        fromSp = false; // derived from the stack pointer, by constants (`belowSp`) or not
        std::string                 truth{};        // a comparison's result: the WGSL `bool` whose `select(0i, 1i, ·)` it is
    };
    enum class Kind { Function, Block, Loop, If };
    struct Frame {
        Kind                     kind;
        std::uint32_t            label;
        std::vector<ValType>     results;
        std::vector<std::string> resultVars;
        std::size_t              height;
        bool                     unreachable = false;
        int                      deadDepth   = 0; // nested constructs opened while unreachable
    };

    std::string                           out{};
    std::vector<Value>                    stack{};
    std::vector<Frame>                    frames{};
    std::map<std::uint32_t, std::int64_t> spLocals{};  // the deepest `belowSp` a local was given; read in program order, so never forgotten
    std::set<std::uint32_t>               unsizedSp{}; // locals given a value derived from the stack pointer by more than constants
    int                                   temps   = 0;
    std::uint32_t                         labels  = 0;
    int                                   depth   = 1;
    std::int64_t                          spDepth = 0; // the deepest the stack pointer was set, relative to its value at entry

    void line(std::string_view text) {
        out.append(static_cast<std::size_t>(depth) * 4, ' ');
        out.append(text);
        out.push_back('\n');
    }
    Value pop() {
        if (stack.empty()) {
            throw Refusal{"value stack underflow"};
        }
        Value v = std::move(stack.back());
        stack.pop_back();
        return v;
    }
    void push(ValType type, std::string expr) {
        const std::string name = std::format("t{}", temps++);
        line(std::format("let {}: {} = {};", name, wgslType(type), expr));
        stack.push_back(Value{name, type});
    }
    static std::string u(const Value& v) { return std::format("bitcast<u32>({})", v.expr); }
    static std::string addressOf(const Value& base, std::uint32_t offset) { return offset == 0 ? u(base) : std::format("({} + {}u)", u(base), offset); }
    static std::string asI32(std::string e) { return std::format("bitcast<i32>({})", e); }
    /// a comparison keeps its `bool`, so a branch on it tests the `bool` instead of re-comparing `select(0i, 1i, c)` with zero
    /// the call arguments [first, count) popped off the stack, as a comma-separated list in call order
    std::string popArguments(std::size_t first, std::size_t count) {
        std::string args;
        for (std::size_t a = first; a < count; ++a) {
            args = (a + 1 == count ? "" : ", ") + pop().expr + args;
        }
        return args;
    }
    void pushTruth(std::string condition) {
        const std::string name = std::format("t{}", temps++);
        line(std::format("let {}: bool = {};", name, condition));
        stack.push_back(Value{.expr = std::format("select(0i, 1i, {})", name), .type = ValType::i32, .truth = name});
    }
    static std::string isTrue(const Value& v) { return v.truth.empty() ? std::format("{} != 0i", v.expr) : v.truth; }

    Frame& current() { return frames.back(); }

    void openFrame(Kind kind, std::vector<ValType> results) {
        Frame f{.kind = kind, .label = ++labels, .results = std::move(results), .resultVars = {}, .height = stack.size()};
        for (ValType t : f.results) {
            const std::string var = std::format("r{}", temps++);
            line(std::format("var {}: {};", var, wgslType(t)));
            f.resultVars.push_back(var);
        }
        frames.push_back(std::move(f));
    }

    // a branch to a loop re-enters it and carries its (absent) parameters; the end of any construct carries its results
    void assignResults(const Frame& target, bool keep, bool atEnd = false) {
        const std::size_t n = (target.kind == Kind::Loop && !atEnd) ? 0 : target.results.size();
        if (stack.size() < n) {
            throw Refusal{"branch without its values"};
        }
        for (std::size_t k = 0; k < n; ++k) {
            line(std::format("{} = {};", target.resultVars[k], stack[stack.size() - n + k].expr));
        }
        if (!keep) {
            stack.resize(stack.size() - n);
        }
    }

    // after a nested construct, or a br_table: hand a pending branch to the frame it targets
    void dispatchPending() {
        const Frame& p = current();
        if (p.kind == Kind::Function) {
            return; // a branch to the function itself is emitted as `return`
        }
        line(std::format("if (tgt != 0u) {{ if (tgt == {}u) {{ tgt = 0u; {} }} else {{ break; }} }}", p.label, p.kind == Kind::Loop ? "continue;" : "break;"));
    }

    void branch(std::uint32_t relativeDepth, bool keepValues, bool insideSwitch) {
        if (relativeDepth >= frames.size()) {
            throw Refusal{"branch depth out of range"};
        }
        const Frame& target = frames[frames.size() - 1 - relativeDepth];
        if (target.kind == Kind::Function) {
            emitReturn(keepValues);
            return;
        }
        assignResults(target, keepValues);
        if (relativeDepth == 0 && !insideSwitch) {
            line(target.kind == Kind::Loop ? "continue;" : "break;");
        } else {
            line(std::format("tgt = {}u;", target.label));
            if (!insideSwitch) {
                line("break;");
            }
        }
    }

    void emitReturn(bool keepValues) {
        const FuncType& type = module.types[module.functions[index].typeIndex];
        if (type.results.empty()) {
            line("return;");
        } else {
            line(std::format("return {};", stack.back().expr));
            if (!keepValues) {
                stack.pop_back();
            }
        }
    }

    void markUnreachable() {
        current().unreachable = true;
        current().deadDepth   = 0;
    }

    std::vector<ValType> blockResults(Reader& r) {
        const std::int64_t type = r.sleb();
        if (type == -64) { // 0x40: empty
            return {};
        }
        if (type < 0) {
            return {static_cast<ValType>(static_cast<std::uint8_t>(type & 0x7F))};
        }
        const FuncType& ft = module.types.at(static_cast<std::size_t>(type));
        if (!ft.params.empty()) {
            throw Refusal{"blocks with parameters are not translated"};
        }
        return ft.results;
    }

    void compare(std::string_view op, bool isUnsigned) {
        const Value b = pop();
        const Value a = pop();
        pushTruth(isUnsigned ? std::format("{} {} {}", u(a), op, u(b)) : std::format("{} {} {}", a.expr, op, b.expr));
    }
    /// `fmt` names the address {0} and the word load {1}: `ldw` where the compiler promises 4-byte alignment, else `ld32`, which may straddle two words
    void load(Reader& r, ValType type, std::string_view fmt) {
        const std::uint32_t    align    = r.u32(); // log2 of the alignment the compiler promises
        const std::uint32_t    offset   = r.u32();
        const Value            base     = pop();
        const std::string      address  = addressOf(base, offset);
        const std::string_view wordLoad = align >= 2 ? "ldw" : "ld32";
        push(type, std::vformat(fmt, std::make_format_args(address, wordLoad)));
    }
    /// `fmt` names the operands {0} and {1}
    void binaryOp(ValType type, std::string_view fmt) {
        const Value b = pop();
        const Value a = pop();
        push(type, std::vformat(fmt, std::make_format_args(a.expr, b.expr)));
        stack.back().fromSp = a.fromSp || b.fromSp;
    }
    void i64Compare(std::string_view fmt) {
        const Value b = pop();
        const Value a = pop();
        pushTruth(std::vformat(fmt, std::make_format_args(a.expr, b.expr)));
    }
    void unary(ValType type, std::string_view fmt) {
        const Value a = pop();
        push(type, std::vformat(fmt, std::make_format_args(a.expr)));
        stack.back().fromSp = a.fromSp;
    }

    // skips one instruction's immediates while the code is unreachable, tracking nested constructs
    bool skipDead(std::uint8_t op, Reader& r) {
        Frame& f = current();
        switch (op) {
        case 0x02:
        case 0x03:
        case 0x04:
            r.sleb();
            ++f.deadDepth;
            return true;
        case 0x05:
            if (f.deadDepth > 0) {
                return true;
            }
            return false;
        case 0x0B:
            if (f.deadDepth > 0) {
                --f.deadDepth;
                return true;
            }
            return false;
        default: skipImmediates(op, r); return true;
        }
    }

    static void skipImmediates(std::uint8_t op, Reader& r) {
        switch (op) {
        case 0x0C:
        case 0x0D:
        case 0x10:
        case 0x20:
        case 0x21:
        case 0x22:
        case 0x23:
        case 0x24:
        case 0x25:
        case 0x26:
        case 0xD2: r.u32(); break;
        case 0x0E:
            for (std::uint32_t n = r.u32() + 1; n > 0; --n) {
                r.u32();
            }
            break;
        case 0x11:
            r.u32();
            r.u32();
            break;
        case 0x1C:
            for (std::uint32_t n = r.u32(); n > 0; --n) {
                r.byte();
            }
            break;
        case 0x3F:
        case 0x40:
        case 0xD0: r.byte(); break;
        case 0x41: r.sleb(); break;
        case 0x42: r.sleb(); break;
        case 0x43: r.take(4); break;
        case 0x44: r.take(8); break;
        case 0xFC: {
            const std::uint32_t sub = r.u32();
            if (sub == 8) {
                r.u32();
                r.byte();
            } else if (sub == 9 || sub == 13 || sub == 15 || sub == 16 || sub == 17) {
                r.u32();
            } else if (sub == 10 || sub == 12 || sub == 14) {
                r.u32();
                r.u32();
            } else if (sub == 11) {
                r.byte();
            }
            break;
        }
        case 0xFD: throw Refusal{"SIMD (0xFD) is not translated"};
        case 0xFE:
            if (r.u32() == 0x03) {
                r.byte(); // atomic.fence
            } else {
                r.u32();
                r.u32();
            }
            break;
        default:
            if (op >= 0x28 && op <= 0x3E) {
                r.u32();
                r.u32();
            }
            break;
        }
    }

    std::string emit() {
        const Function& fn   = module.functions[index];
        const FuncType& type = module.types[fn.typeIndex];
        if (type.results.size() > 1) {
            throw Refusal{"multi-value results are not translated"};
        }
        std::string signature;
        for (std::size_t p = 0; p < type.params.size(); ++p) {
            signature += std::format("{}p{}: {}", p == 0 ? "" : ", ", p, wgslType(type.params[p]));
        }
        std::string header = std::format("fn f{}({}){} {{\n", index, signature, type.results.empty() ? "" : std::format(" -> {}", wgslType(type.results[0])));
        for (std::size_t p = 0; p < type.params.size(); ++p) {
            line(std::format("var l{}: {} = p{};", p, wgslType(type.params[p]), p));
        }
        for (std::size_t l = 0; l < fn.locals.size(); ++l) {
            line(std::format("var l{}: {};", type.params.size() + l, wgslType(fn.locals[l])));
        }
        line("var tgt: u32 = 0u;");
        frames.push_back(Frame{.kind = Kind::Function, .label = 0, .results = type.results, .resultVars = {}, .height = 0});

        Reader r{fn.body};
        while (!r.atEnd()) {
            const std::uint8_t op = r.byte();
            if (current().unreachable && skipDead(op, r)) {
                continue;
            }
            ++instructions;
            instruction(op, r, type);
            if (frames.empty()) {
                break;
            }
        }
        if (!type.results.empty()) {
            line(std::format("return {};", zeroOf(type.results[0]))); // after a trailing construct; never reached
        }
        return header + out + "}\n";
    }

    void closeConstruct() {
        Frame f = std::move(frames.back());
        frames.pop_back();
        if (!f.unreachable) {
            assignResults(f, false, true);
            line("break;");
        }
        stack.resize(f.height);
        --depth;
        line("}");
        if (f.kind == Kind::If) {
            --depth;
            line("}");
        }
        for (const std::string& var : f.resultVars) {
            stack.push_back(Value{var, f.results[static_cast<std::size_t>(&var - f.resultVars.data())]});
        }
        dispatchPending();
    }

    void instruction(std::uint8_t op, Reader& r, const FuncType& type) {
        switch (op) {
        case 0x00: // a trap has no WGSL equivalent: leave the function, with a zero value where one is due
            line(type.results.empty() ? std::string("return;") : std::format("return {};", zeroOf(type.results[0])));
            markUnreachable();
            break;
        case 0x01: break;
        case 0x02:
        case 0x03: {
            const auto results = blockResults(r);
            openFrame(op == 0x02 ? Kind::Block : Kind::Loop, results);
            line("loop {");
            ++depth;
            break;
        }
        case 0x04: {
            const auto  results   = blockResults(r);
            const Value condition = pop();
            openFrame(Kind::If, results);
            line(std::format("if ({}) {{", isTrue(condition)));
            ++depth;
            line("loop {");
            ++depth;
            break;
        }
        case 0x05: {
            Frame& f = current();
            if (!f.unreachable) {
                assignResults(f, false);
                line("break;");
            }
            stack.resize(f.height);
            f.unreachable = false;
            depth -= 2;
            line("}");
            line("} else {");
            ++depth;
            line("loop {");
            ++depth;
            break;
        }
        case 0x0B:
            if (frames.size() == 1) { // function end
                if (!current().unreachable && !type.results.empty()) {
                    emitReturn(false);
                }
                frames.pop_back();
            } else {
                closeConstruct();
            }
            break;
        case 0x0C:
            branch(r.u32(), false, false);
            markUnreachable();
            break;
        case 0x0D: {
            const std::uint32_t target    = r.u32();
            const Value         condition = pop();
            line(std::format("if ({}) {{", isTrue(condition)));
            ++depth;
            branch(target, true, false);
            --depth;
            line("}");
            break;
        }
        case 0x0E: {
            std::vector<std::uint32_t> targets(r.u32() + 1);
            for (auto& t : targets) {
                t = r.u32();
            }
            const Value selector = pop();
            line(std::format("switch ({}) {{", u(selector)));
            ++depth;
            for (std::size_t k = 0; k < targets.size(); ++k) {
                line(k + 1 == targets.size() ? "default: {" : std::format("case {}u: {{", k));
                ++depth;
                const std::size_t height = stack.size();
                branch(targets[k], true, true);
                stack.resize(height);
                --depth;
                line("}");
            }
            --depth;
            line("}");
            const Frame& f = current();
            if (f.kind == Kind::Function) {
                emitReturn(false);
            } else {
                dispatchPending();
            }
            markUnreachable();
            break;
        }
        case 0x0F:
            emitReturn(false);
            markUnreachable();
            break;
        case 0x10: {
            const std::uint32_t callee = r.u32();
            const Function&     target = module.functions.at(callee);
            if (const auto builtin = std::ranges::find(kLibmBuiltins, target.name, &LibmBuiltin::libm); builtin != kLibmBuiltins.end()) {
                const FuncType&   ct   = module.types[target.typeIndex];
                const std::string args = popArguments(0UZ, ct.params.size());
                push(ValType::f32, std::format("{}({})", builtin->wgsl, args));
                break;
            }
            const FuncType& ct = module.types[target.typeIndex];
            if (isLandingPadRuntime(target.name)) {
                for (std::size_t a = 0; a < ct.params.size(); ++a) {
                    pop();
                }
                if (!ct.results.empty()) {
                    push(ct.results[0], zeroOf(ct.results[0]));
                }
                break;
            }
            // Emscripten's exception lowering calls `invoke_<sig>(target, args...)`, so that JavaScript can catch a
            // throw and set `__THREW__`; GPU code cannot throw, so the target is called directly
            const bool viaInvoke = target.imported && target.importName.starts_with("env.invoke_");
            if (target.imported && !viaInvoke) {
                throw Refusal{std::format("calls the import '{}'", target.importName)};
            }
            const std::string args   = popArguments(viaInvoke ? 1UZ : 0UZ, ct.params.size());
            std::uint32_t     called = callee;
            if (viaInvoke) {
                const Value pointer = pop();
                const auto  entry   = pointer.constant ? module.tableToFunction.find(static_cast<std::uint32_t>(*pointer.constant)) : module.tableToFunction.end();
                if (entry == module.tableToFunction.end()) {
                    throw Refusal{std::format("'{}' calls through a function pointer that is not a constant", target.name)};
                }
                called                  = entry->second;
                const Function& direct  = module.functions.at(called);
                const FuncType& dt      = module.types[direct.typeIndex];
                const bool      matches = !direct.imported && dt.results == ct.results && std::ranges::equal(dt.params, ct.params | std::views::drop(1));
                if (!matches) {
                    throw Refusal{std::format("'{}' calls '{}', an import or a function of another type", target.name, demangle(direct.name))};
                }
            }
            ++callees[called];
            if (ct.results.size() > 1) {
                throw Refusal{"multi-value calls are not translated"};
            }
            if (ct.results.empty()) {
                line(std::format("f{}({});", called, args));
            } else {
                push(ct.results[0], std::format("f{}({})", called, args));
            }
            break;
        }
        case 0x11: throw Refusal{"an indirect call (function pointer or virtual) has no WGSL equivalent"};
        case 0x1A: pop(); break;
        case 0x1B:
        case 0x1C: {
            if (op == 0x1C) {
                for (std::uint32_t n = r.u32(); n > 0; --n) {
                    r.byte();
                }
            }
            const Value c = pop();
            const Value b = pop();
            const Value a = pop();
            push(a.type, std::format("select({}, {}, {})", b.expr, a.expr, isTrue(c)));
            stack.back().fromSp = a.fromSp || b.fromSp;
            break;
        }
        case 0x20: {
            const std::uint32_t local = r.u32();
            push(localType(local, type), std::format("l{}", local));
            if (const auto held = spLocals.find(local); held != spLocals.end() && !unsizedSp.contains(local)) {
                stack.back().belowSp = held->second;
            }
            stack.back().fromSp = spLocals.contains(local) || unsizedSp.contains(local);
            break;
        }
        case 0x21:
        case 0x22: {
            const std::uint32_t local = r.u32();
            const Value         value = op == 0x21 ? pop() : stack.back();
            line(std::format("l{} = {};", local, value.expr));
            if (value.belowSp) {
                spLocals[local] = std::max(spLocals[local], *value.belowSp);
            } else if (value.fromSp) {
                unsizedSp.insert(local);
            }
            break;
        }
        case 0x23: {
            const std::uint32_t g  = r.u32();
            const Global&       gl = module.globals.at(g);
            if (gl.isMutable && gl.name == "GOT.data.internal.__THREW__") {
                readsThrew = true;
                push(gl.type, "gr_threw");
            } else if (gl.isMutable) {
                if (g != module.stackPointer) {
                    throw Refusal{std::format("reads the mutable global '{}' (thread-local state)", gl.name.empty() ? std::to_string(g) : gl.name)};
                }
                push(gl.type, "sp");
                stack.back().belowSp = spDepth;
                stack.back().fromSp  = true;
            } else if (gl.imported) {
                throw Refusal{std::format("reads the imported global '{}', whose value is known only when the module loads", gl.name)};
            } else {
                push(gl.type, gl.type == ValType::f32 ? std::format("bitcast<f32>(0x{:08x}u)", static_cast<std::uint32_t>(gl.init)) : i32Literal(gl.init));
            }
            break;
        }
        case 0x24: {
            const std::uint32_t g = r.u32();
            if (g != module.stackPointer) {
                throw Refusal{std::format("writes the global '{}'", module.globals.at(g).name)};
            }
            const Value value = pop();
            if (!value.belowSp) {
                throw Refusal{"sets the stack pointer to a value it cannot size (a variable-length stack allocation)"};
            }
            spDepth = std::max(spDepth, *value.belowSp); // the deepest so far: an epilogue in one branch does not undo a prologue
            line(std::format("sp = {};", value.expr));
            break;
        }
        case 0x28: load(r, ValType::i32, "bitcast<i32>({1}({0}))"); break;
        case 0x2A: load(r, ValType::f32, "bitcast<f32>({1}({0}))"); break;
        case 0x2C: load(r, ValType::i32, "(bitcast<i32>(ld8({0}) << 24u) >> 24u)"); break;
        case 0x2D: load(r, ValType::i32, "bitcast<i32>(ld8({0}))"); break;
        case 0x2E: load(r, ValType::i32, "(bitcast<i32>(ld16({0}) << 16u) >> 16u)"); break;
        case 0x2F: load(r, ValType::i32, "bitcast<i32>(ld16({0}))"); break;
        case 0x36:
        case 0x38:
        case 0x3A:
        case 0x3B: {
            const std::uint32_t align  = r.u32(); // log2 of the alignment the compiler promises
            const std::uint32_t offset = r.u32();
            const Value         value  = pop();
            const Value         base   = pop();
            const std::string   a      = addressOf(base, offset);
            const std::string   bits   = value.type == ValType::f32 ? std::format("bitcast<u32>({})", value.expr) : u(value);
            const char*         helper = op == 0x3A ? "st8" : op == 0x3B ? "st16" : align >= 2 ? "stw" : "st32";
            if (op == 0x3A || op == 0x3B || align < 2) { // a lane, or a word that may straddle two
                writesLanes = true;
            }
            line(std::format("{}({}, {});", helper, a, bits));
            break;
        }
        case 0x29: load(r, ValType::i64, "vec2<u32>({1}({0}), {1}({0} + 4u))"); break;
        case 0x30: load(r, ValType::i64, "i64_from_i32(bitcast<i32>(ld8({0}) << 24u) >> 24u)"); break;
        case 0x31: load(r, ValType::i64, "vec2<u32>(ld8({0}), 0u)"); break;
        case 0x32: load(r, ValType::i64, "i64_from_i32(bitcast<i32>(ld16({0}) << 16u) >> 16u)"); break;
        case 0x33: load(r, ValType::i64, "vec2<u32>(ld16({0}), 0u)"); break;
        case 0x34: load(r, ValType::i64, "i64_from_i32(bitcast<i32>({1}({0})))"); break;
        case 0x35: load(r, ValType::i64, "vec2<u32>({1}({0}), 0u)"); break;
        case 0x37:
        case 0x3C:
        case 0x3D:
        case 0x3E: {
            const std::uint32_t align  = r.u32();
            const std::uint32_t offset = r.u32();
            const Value         value  = pop();
            const Value         base   = pop();
            const std::string   a      = addressOf(base, offset);
            if (op == 0x37) {
                writesLanes                 = writesLanes || align < 2;
                const char* const wordStore = align >= 2 ? "stw" : "st32";
                line(std::format("{2}({0}, {1}.x); {2}({0} + 4u, {1}.y);", a, value.expr, wordStore));
            } else {
                writesLanes = writesLanes || op != 0x3E || align < 2;
                line(std::format("{}({}, {}.x);", op == 0x3C ? "st8" : op == 0x3D ? "st16" : align >= 2 ? "stw" : "st32", a, value.expr));
            }
            break;
        }
        case 0x2B:
        case 0x39: throw Refusal{"a double-precision (f64) memory access has no WGSL equivalent"};
        case 0x3F:
        case 0x40: throw Refusal{"memory.size/memory.grow inside a kernel"};
        case 0x41: {
            const std::int64_t value = r.sleb();
            push(ValType::i32, i32Literal(value));
            stack.back().constant = value;
            break;
        }
        case 0x42: {
            const auto bits = static_cast<std::uint64_t>(r.sleb());
            push(ValType::i64, std::format("vec2<u32>(0x{:08x}u, 0x{:08x}u)", bits & 0xFFFFFFFFU, bits >> 32U));
            break;
        }
        case 0x43: {
            const std::uint32_t bits = r.fixed32();
            // WGSL rejects a constant expression that yields inf or NaN, so such a pattern is formed at run time
            // opaque to the compiler, which would otherwise fold it and reassociate around it (qa_Wasm2Wgsl, sin/cos case);
            // it also keeps inf and NaN out of constant expressions, which WGSL rejects
            push(ValType::f32, std::format("bitcast<f32>(0x{:08x}u + launch.zero)", bits));
            break;
        }
        case 0x44: throw Refusal{"double precision (f64) constant"};
        case 0x45: {
            const Value a = pop();
            pushTruth(a.truth.empty() ? std::format("{} == 0i", a.expr) : std::format("!{}", a.truth));
            break;
        }
        case 0x46: compare("==", false); break;
        case 0x47: compare("!=", false); break;
        case 0x48: compare("<", false); break;
        case 0x49: compare("<", true); break;
        case 0x4A: compare(">", false); break;
        case 0x4B: compare(">", true); break;
        case 0x4C: compare("<=", false); break;
        case 0x4D: compare("<=", true); break;
        case 0x4E: compare(">=", false); break;
        case 0x4F: compare(">=", true); break;
        case 0x5B: compare("==", false); break;
        case 0x5C: compare("!=", false); break;
        case 0x5D: compare("<", false); break;
        case 0x5E: compare(">", false); break;
        case 0x5F: compare("<=", false); break;
        case 0x60: compare(">=", false); break;
        case 0x67: unary(ValType::i32, "bitcast<i32>(countLeadingZeros(bitcast<u32>({})))"); break;
        case 0x68: unary(ValType::i32, "bitcast<i32>(countTrailingZeros(bitcast<u32>({})))"); break;
        case 0x69: unary(ValType::i32, "bitcast<i32>(countOneBits(bitcast<u32>({})))"); break;
        case 0x6A: {
            const auto below = spOffsetBy(-1);
            binaryOp(ValType::i32, "{0} + {1}");
            sizeFrame(below);
            break;
        }
        case 0x6B: {
            const auto below = spOffsetBy(+1);
            binaryOp(ValType::i32, "{0} - {1}");
            sizeFrame(below);
            break;
        }
        case 0x6C: binaryOp(ValType::i32, "{0} * {1}"); break;
        case 0x6D: binaryOp(ValType::i32, "{0} / {1}"); break;
        case 0x6E: binaryOp(ValType::i32, "bitcast<i32>(bitcast<u32>({0}) / bitcast<u32>({1}))"); break;
        case 0x6F: binaryOp(ValType::i32, "{0} % {1}"); break;
        case 0x70: binaryOp(ValType::i32, "bitcast<i32>(bitcast<u32>({0}) % bitcast<u32>({1}))"); break;
        case 0x71: {
            // `sp - N` realigned by `& -K`, for an over-aligned local, can reach K - 1 bytes further
            const Value&                      mask  = stack.back();
            const Value&                      base  = stack[stack.size() - 2];
            const std::optional<std::int64_t> below = base.belowSp && mask.constant && *mask.constant < 0 ? std::optional{*base.belowSp - *mask.constant - 1} : std::nullopt;
            binaryOp(ValType::i32, "{0} & {1}");
            sizeFrame(below);
            break;
        }
        case 0x72: binaryOp(ValType::i32, "{0} | {1}"); break;
        case 0x73: binaryOp(ValType::i32, "{0} ^ {1}"); break;
        case 0x74: binaryOp(ValType::i32, "{0} << (bitcast<u32>({1}) & 31u)"); break;
        case 0x75: binaryOp(ValType::i32, "{0} >> (bitcast<u32>({1}) & 31u)"); break;
        case 0x76: binaryOp(ValType::i32, "bitcast<i32>(bitcast<u32>({0}) >> (bitcast<u32>({1}) & 31u))"); break;
        case 0x77: binaryOp(ValType::i32, "bitcast<i32>((bitcast<u32>({0}) << (bitcast<u32>({1}) & 31u)) | (bitcast<u32>({0}) >> ((32u - (bitcast<u32>({1}) & 31u)) & 31u)))"); break;
        case 0x78: binaryOp(ValType::i32, "bitcast<i32>((bitcast<u32>({0}) >> (bitcast<u32>({1}) & 31u)) | (bitcast<u32>({0}) << ((32u - (bitcast<u32>({1}) & 31u)) & 31u)))"); break;
        case 0x8B: unary(ValType::f32, "abs({})"); break;
        case 0x8C: unary(ValType::f32, "-({})"); break;
        case 0x8D: unary(ValType::f32, "ceil({})"); break;
        case 0x8E: unary(ValType::f32, "floor({})"); break;
        case 0x8F: unary(ValType::f32, "trunc({})"); break;
        case 0x90: unary(ValType::f32, "round({})"); break;
        case 0x91: unary(ValType::f32, "sqrt({})"); break;
        case 0x92: binaryOp(ValType::f32, "{0} + {1}"); break;
        case 0x93: binaryOp(ValType::f32, "{0} - {1}"); break;
        case 0x94: binaryOp(ValType::f32, "{0} * {1}"); break;
        case 0x95: binaryOp(ValType::f32, "{0} / {1}"); break;
        case 0x96: binaryOp(ValType::f32, "min({0}, {1})"); break;
        case 0x97: binaryOp(ValType::f32, "max({0}, {1})"); break;
        case 0x98: binaryOp(ValType::f32, "bitcast<f32>((bitcast<u32>({0}) & 0x7fffffffu) | (bitcast<u32>({1}) & 0x80000000u))"); break;
        case 0xA8: unary(ValType::i32, "i32({})"); break;
        case 0xA9: unary(ValType::i32, "bitcast<i32>(u32({}))"); break;
        case 0xB2: unary(ValType::f32, "f32({})"); break;
        case 0xB3: unary(ValType::f32, "f32(bitcast<u32>({}))"); break;
        case 0xBC: unary(ValType::i32, "bitcast<i32>({})"); break;
        case 0xBE: unary(ValType::f32, "bitcast<f32>({})"); break;
        case 0xC0: unary(ValType::i32, "(({} << 24u) >> 24u)"); break;
        case 0xC1: unary(ValType::i32, "(({} << 16u) >> 16u)"); break;
        case 0xFC: {
            const std::uint32_t sub = r.u32();
            if (sub == 0) {
                unary(ValType::i32, "i32({})");
            } else if (sub == 1) {
                unary(ValType::i32, "bitcast<i32>(u32({}))");
            } else if (sub == 4) {
                unary(ValType::i64, "i64_from_f32({})");
            } else if (sub == 5) {
                unary(ValType::i64, "i64_from_f32(max({}, 0.0f))");
            } else if (sub == 10) {
                r.byte();
                r.byte();
                const Value n = pop();
                const Value s = pop();
                const Value d = pop();
                line(std::format("memmove({}, {}, {});", u(d), u(s), u(n)));
                writesLanes = true;
            } else if (sub == 11) {
                r.byte();
                const Value n = pop();
                const Value v = pop();
                const Value d = pop();
                line(std::format("memfill({}, {}, {});", u(d), u(v), u(n)));
                writesLanes = true;
            } else {
                throw Refusal{std::format("unsupported operation 0xFC {}", sub)};
            }
            break;
        }
        case 0xFD: throw Refusal{"SIMD (0xFD) is not translated"};
        case 0xFE: { // threads proposal: word-sized atomics on the atomic heap; wasm requires them naturally aligned
            const std::uint32_t sub = r.u32();
            if (sub == 0x03) { // atomic.fence: every heap access is already an atomic one
                r.byte();
                break;
            }
            r.u32(); // alignment
            const std::uint32_t                                                 offset  = r.u32();
            const auto                                                          address = [&](const Value& base) { return addressOf(base, offset); };
            constexpr std::array<std::pair<std::uint32_t, std::string_view>, 6> kRmw{{{0x1E, "atomicAdd"}, {0x25, "atomicSub"}, {0x2C, "atomicAnd"}, {0x33, "atomicOr"}, {0x3A, "atomicXor"}, {0x41, "atomicExchange"}}};
            writesLanes = true;
            if (sub == 0x10) {
                const Value base = pop();
                push(ValType::i32, asI32(std::format("atomicLoad(&heap[{} >> 2u])", address(base))));
            } else if (sub == 0x17) {
                const Value value = pop();
                const Value base  = pop();
                line(std::format("atomicStore(&heap[{} >> 2u], {});", address(base), u(value)));
            } else if (const auto rmw = std::ranges::find(kRmw, sub, &std::pair<std::uint32_t, std::string_view>::first); rmw != kRmw.end()) {
                const Value value = pop();
                const Value base  = pop();
                push(ValType::i32, asI32(std::format("{}(&heap[{} >> 2u], {})", rmw->second, address(base), u(value))));
            } else if (sub == 0x48) {
                const Value replacement = pop();
                const Value expected    = pop();
                const Value base        = pop();
                push(ValType::i32, asI32(std::format("gr_cmpxchg({} >> 2u, {}, {})", address(base), u(expected), u(replacement))));
            } else if (sub == 0x11) {
                const Value base = pop();
                const auto  a    = address(base);
                push(ValType::i64, std::format("vec2<u32>(atomicLoad(&heap[{0} >> 2u]), atomicLoad(&heap[({0} >> 2u) + 1u]))", a)); // two words: can tear
            } else if (sub == 0x18) {
                const Value value = pop();
                const Value base  = pop();
                line(std::format("atomicStore(&heap[{0} >> 2u], {1}.x); atomicStore(&heap[({0} >> 2u) + 1u], {1}.y);", address(base), value.expr)); // two words: can tear
            } else if (sub == 0x1F) {                                                                                                               // add: atomic per word with the carry propagated, so concurrent sums stay exact
                const Value value = pop();
                const Value base  = pop();
                const auto  a     = address(base);
                const auto  low   = std::format("t{}", temps++);
                line(std::format("let {0}: u32 = atomicAdd(&heap[{1} >> 2u], {2}.x);", low, a, value.expr));
                push(ValType::i64, std::format("vec2<u32>({0}, atomicAdd(&heap[({1} >> 2u) + 1u], {2}.y + select(0u, 1u, {0} + {2}.x < {0})))", low, a, value.expr)); // the returned high word may lag a concurrent carry
            } else if (sub == 0x2D || sub == 0x34 || sub == 0x3B) {                                                                                                   // and, or, xor: word by word is exact
                const Value       value    = pop();
                const Value       base     = pop();
                const auto        a        = address(base);
                const std::string atomicOp = sub == 0x2D ? "atomicAnd" : sub == 0x34 ? "atomicOr" : "atomicXor";
                push(ValType::i64, std::format("vec2<u32>({0}(&heap[{1} >> 2u], {2}.x), {0}(&heap[({1} >> 2u) + 1u], {2}.y))", atomicOp, a, value.expr));
            } else {
                throw Refusal{std::format("atomic operation 0xFE 0x{:02x} (wait/notify, sub-word or 64-bit) is not translated", sub)};
            }
            break;
        }
        case 0x50: {
            const Value a = pop();
            pushTruth(std::format("all({} == vec2<u32>(0u, 0u))", a.expr));
            break;
        }
        case 0x51: i64Compare("all({0} == {1})"); break;
        case 0x52: i64Compare("any({0} != {1})"); break;
        case 0x53: i64Compare("i64_lt_s({0}, {1})"); break;
        case 0x54: i64Compare("i64_lt_u({0}, {1})"); break;
        case 0x55: i64Compare("i64_lt_s({1}, {0})"); break;
        case 0x56: i64Compare("i64_lt_u({1}, {0})"); break;
        case 0x57: i64Compare("!i64_lt_s({1}, {0})"); break;
        case 0x58: i64Compare("!i64_lt_u({1}, {0})"); break;
        case 0x59: i64Compare("!i64_lt_s({0}, {1})"); break;
        case 0x5A: i64Compare("!i64_lt_u({0}, {1})"); break;
        case 0x79: unary(ValType::i64, "i64_clz({})"); break;
        case 0x7A: unary(ValType::i64, "i64_ctz({})"); break;
        case 0x7B: unary(ValType::i64, "vec2<u32>(countOneBits({0}.x) + countOneBits({0}.y), 0u)"); break;
        case 0x7C: binaryOp(ValType::i64, "i64_add({0}, {1})"); break;
        case 0x7D: binaryOp(ValType::i64, "i64_sub({0}, {1})"); break;
        case 0x7E: binaryOp(ValType::i64, "i64_mul({0}, {1})"); break;
        case 0x7F: binaryOp(ValType::i64, "i64_div_s({0}, {1})"); break;
        case 0x80: binaryOp(ValType::i64, "i64_divrem_u({0}, {1}).q"); break;
        case 0x81: binaryOp(ValType::i64, "i64_rem_s({0}, {1})"); break;
        case 0x82: binaryOp(ValType::i64, "i64_divrem_u({0}, {1}).r"); break;
        case 0x83: binaryOp(ValType::i64, "({0} & {1})"); break;
        case 0x84: binaryOp(ValType::i64, "({0} | {1})"); break;
        case 0x85: binaryOp(ValType::i64, "({0} ^ {1})"); break;
        case 0x86: binaryOp(ValType::i64, "i64_shl({0}, {1}.x)"); break;
        case 0x87: binaryOp(ValType::i64, "i64_shr_s({0}, {1}.x)"); break;
        case 0x88: binaryOp(ValType::i64, "i64_shr_u({0}, {1}.x)"); break;
        case 0x89: binaryOp(ValType::i64, "(i64_shl({0}, {1}.x) | i64_shr_u({0}, (64u - ({1}.x & 63u)) & 63u))"); break;
        case 0x8A: binaryOp(ValType::i64, "(i64_shr_u({0}, {1}.x) | i64_shl({0}, (64u - ({1}.x & 63u)) & 63u))"); break;
        case 0xA7: unary(ValType::i32, "bitcast<i32>({}.x)"); break;
        case 0xAC: unary(ValType::i64, "i64_from_i32({})"); break;
        case 0xAD: unary(ValType::i64, "vec2<u32>(bitcast<u32>({}), 0u)"); break;
        case 0xAE: unary(ValType::i64, "i64_from_f32({})"); break;
        case 0xAF: unary(ValType::i64, "i64_from_f32(max({}, 0.0f))"); break;
        case 0xB4: unary(ValType::f32, "i64_to_f32_s({})"); break;
        case 0xB5: unary(ValType::f32, "i64_to_f32_u({})"); break;
        case 0xC2: unary(ValType::i64, "i64_from_i32((bitcast<i32>({}.x) << 24u) >> 24u)"); break;
        case 0xC3: unary(ValType::i64, "i64_from_i32((bitcast<i32>({}.x) << 16u) >> 16u)"); break;
        case 0xC4: unary(ValType::i64, "i64_from_i32(bitcast<i32>({}.x))"); break;
        default:
            if ((op >= 0x61 && op <= 0x66) || (op >= 0x99 && op <= 0xA6) || op == 0xAA || op == 0xAB || op == 0xB0 || op == 0xB1 || (op >= 0xB6 && op <= 0xBB) || op == 0xBD || op == 0xBF) {
                throw Refusal{std::format("double-precision (f64) operation 0x{:02x} has no WGSL equivalent", op)};
            }
            throw Refusal{std::format("unsupported opcode 0x{:02x}", op)};
        }
    }

    /// a leaf function addresses its frame at `sp - N` without moving sp, so every such address counts, not only a `global.set`
    void sizeFrame(std::optional<std::int64_t> below) {
        stack.back().belowSp = below;
        if (below) {
            frameBytes = std::max(frameBytes, static_cast<std::uint32_t>(std::max<std::int64_t>(0, *below)));
        }
    }
    /// `sp ± N` (`sign` +1 for `i32.sub`, -1 for `i32.add`) on the two operands about to be popped: the result's `belowSp`
    std::optional<std::int64_t> spOffsetBy(int sign) const {
        const Value& b = stack.back();
        const Value& a = stack[stack.size() - 2];
        if (a.belowSp && b.constant) {
            return *a.belowSp + sign * *b.constant;
        }
        if (sign < 0 && b.belowSp && a.constant) {
            return *b.belowSp - *a.constant;
        }
        return std::nullopt;
    }

    ValType localType(std::uint32_t local, const FuncType& type) const {
        const Function& fn = module.functions[index];
        return local < type.params.size() ? type.params[local] : fn.locals.at(local - type.params.size());
    }
};

// one word of the heap, and a byte or half-word lane of one: a kernel that writes lanes gets an atomic heap, so work
// items writing neighbouring lanes of one word do not lose each other's updates
constexpr char kPlainHeapText[] = {
#embed "heap_plain.wgsl"
    , '\0'};
constexpr std::string_view kPlainHeap{kPlainHeapText};
constexpr char             kAtomicHeapText[] = {
#embed "heap_atomic.wgsl"
    , '\0'};
constexpr std::string_view kAtomicHeap{kAtomicHeapText};

// the WGSL every kernel starts with, kept as WGSL: `#embed` is C++26, and GCC 15 and Clang take it before
constexpr char kPreludeText[] = {
#embed "prelude.wgsl"
    , '\0'};
constexpr std::string_view kPrelude{kPreludeText};

inline constexpr std::uint32_t kThrewSlotBytes   = 16U; // a work item's `__THREW__` word, at the top of its stack region, kept 16-byte aligned
inline constexpr std::uint32_t kStackMarginBytes = 16U; // below each work item's deepest frame
// drivers inline every call, and a pipeline's build grows faster than its inlined size: NVIDIA's GPU process took tens of
// seconds just below this and was lost well above it
inline constexpr std::uint64_t kMaxInlinedInstructions = 20'000U;

struct Kernel {
    std::uint32_t tableIndex = 0;
    std::string   name{};
    std::string   wgsl{};
    std::string   refusal{};
    std::uint32_t stackBytes          = 0; // per work item, deepest call path
    std::uint64_t inlinedInstructions = 0; // what a driver compiles: it inlines every call
};

inline Kernel translateKernel(const Module& m, std::uint32_t tableIndex, std::uint32_t root) {
    Kernel kernel{.tableIndex = tableIndex, .name = demangle(m.functions[root].name)};
    try {
        const FuncType& rootType = m.types[m.functions[root].typeIndex];
        if (rootType.params.size() != 2 || rootType.params[0] != ValType::i32 || rootType.params[1] != ValType::i32 || !rootType.results.empty()) {
            throw Refusal{"the kernel entry is not (i32 closure, i32 index) -> ()"};
        }
        std::map<std::uint32_t, std::map<std::uint32_t, std::uint32_t>> calls; // caller -> callee -> call sites
        std::map<std::uint32_t, std::string>                            bodies;
        std::map<std::uint32_t, std::uint32_t>                          frames;
        std::map<std::uint32_t, std::uint32_t>                          instructions;
        bool                                                            writesLanes = false;
        bool                                                            readsThrew  = false;
        std::map<std::uint32_t, std::uint32_t>                          calledFrom; // the first caller found: a refusal names its path from the kernel
        std::vector<std::uint32_t>                                      pending{root};
        while (!pending.empty()) {
            const std::uint32_t f = pending.back();
            pending.pop_back();
            if (bodies.contains(f)) {
                continue;
            }
            std::map<std::uint32_t, std::uint32_t> callees;
            try {
                bodies[f] = FunctionEmitter{.module = m, .index = f, .callees = callees, .instructions = instructions[f], .writesLanes = writesLanes, .readsThrew = readsThrew, .frameBytes = frames[f]}.emit();
            } catch (const Refusal& refusal) {
                std::string path = demangle(m.functions[f].name);
                for (auto caller = calledFrom.find(f); caller != calledFrom.end() && caller->second != root; caller = calledFrom.find(caller->second)) {
                    path += std::format("', called from '{}", demangle(m.functions[caller->second].name));
                }
                throw Refusal{std::format("{} (in '{}')", refusal.reason, path)};
            }
            for (const auto& [callee, sites] : callees) {
                calledFrom.try_emplace(callee, f);
                pending.push_back(callee);
            }
            calls[f] = std::move(callees);
        }
        // WGSL forbids recursion: a cycle in the call graph refuses the kernel
        std::map<std::uint32_t, int> state;
        const auto                   visit = [&](auto& self, std::uint32_t f) -> void {
            state[f] = 1;
            for (const auto& [c, sites] : calls[f]) {
                if (state[c] == 1) {
                    throw Refusal{std::format("recursion through '{}'", demangle(m.functions[c].name))};
                }
                if (state[c] == 0) {
                    self(self, c);
                }
            }
            state[f] = 2;
        };
        visit(visit, root);

        std::map<std::uint32_t, std::uint32_t> deepest; // stack bytes of the deepest path from a function, the call graph being acyclic
        const auto                             depthOf = [&](auto& self, std::uint32_t f) -> std::uint32_t {
            if (const auto it = deepest.find(f); it != deepest.end()) {
                return it->second;
            }
            std::uint32_t callee = 0;
            for (const auto& [c, sites] : calls[f]) {
                callee = std::max(callee, self(self, c));
            }
            return deepest[f] = frames[f] + callee;
        };
        std::map<std::uint32_t, std::uint64_t> inlined;
        const auto                             inlinedOf = [&](auto& self, std::uint32_t f) -> std::uint64_t {
            if (const auto it = inlined.find(f); it != inlined.end()) {
                return it->second;
            }
            std::uint64_t total = instructions[f];
            for (const auto& [c, sites] : calls[f]) {
                total += sites * self(self, c);
            }
            return inlined[f] = total;
        };
        kernel.inlinedInstructions = inlinedOf(inlinedOf, root);
        if (kernel.inlinedInstructions > kMaxInlinedInstructions) {
            throw Refusal{std::format("{} wasm instructions once every call is inlined, above the {} that drivers build reliably", kernel.inlinedInstructions, kMaxInlinedInstructions)};
        }
        const std::uint32_t used = depthOf(depthOf, root) + (readsThrew ? kThrewSlotBytes : 0U);
        kernel.stackBytes        = used == 0 ? 0 : (used + kStackMarginBytes + 15U) & ~15U;

        kernel.wgsl              = std::string(kPrelude);
        const std::size_t marker = kernel.wgsl.find("HEAP_ACCESS\n");
        kernel.wgsl.replace(marker, std::string_view("HEAP_ACCESS\n").size(), writesLanes ? kAtomicHeap : kPlainHeap);
        for (const auto& [f, body] : bodies) {
            kernel.wgsl += std::format("\n// {}\n{}", demangle(m.functions[f].name), body);
        }
        kernel.wgsl += std::format(R"(
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {{
    let index = id.x + id.y * groups.x * 64u;
    if (index >= launch.count) {{ return; }}
    sp = bitcast<i32>(launch.stackTop - index * launch.stackBytes);{}
    f{}(bitcast<i32>(launch.closure), bitcast<i32>(index));
}}
)",
            readsThrew ? std::format("\n    gr_threw = sp - {}i;\n    sp = gr_threw;\n    setWord(bitcast<u32>(gr_threw) >> 2u, 0u);", kThrewSlotBytes) : std::string(), root);
    } catch (const Refusal& refusal) {
        kernel.refusal = refusal.reason;
        kernel.wgsl.clear();
    }
    return kernel;
}

} // namespace wasm2wgsl

#endif // WASM2WGSL_WGSL_KERNEL_HPP
