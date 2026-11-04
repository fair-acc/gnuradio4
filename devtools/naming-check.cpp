// naming-check: the naming rules of CLAUDE.md §3, checked on every declaration libclang sees in the given roots.
// Each declaration is classified once (by its cursor kind and context) into one rule; a violation is reported with
// the rule, the expected form and a proposed spelling that is valid and free in its scope. Observations tag what the
// rules leave open, so the counts can settle them. `naming-check --help` lists the options and exit codes.

#include "Libclang.hpp"
#include "NameSpelling.hpp"

#include <clang-c/Index.h>

#include <unistd.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <charconv>
#include <cstddef>
#include <cstdio>
#include <filesystem>
#include <format>
#include <fstream>
#include <iterator>
#include <map>
#include <optional>
#include <print>
#include <ranges>
#include <regex>
#include <set>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

namespace fs = std::filesystem;
using namespace devtools;

const std::regex kBlockBase{"(^|::)Block<"};
const std::regex kPortType{"Port(In|Out)?<|PortIn|PortOut"};
const std::regex kBoostSuite{"boost::ut::(\\w+::)?suite<"};

// a lowercase macro that mirrors or extends a keyword or attribute (`forceinline`, `noinline`) is accepted (A13)
const std::set<std::string_view> kKeywordLikeWords{"inline", "noreturn", "nodiscard", "likely", "unlikely", "fallthrough", "restrict", "unused", "deprecated", "constexpr", "consteval"};

// a trailing `_` that only dodges a keyword (`long_`) is accepted
const std::set<std::string_view> kKeywords{"alignas", "alignof", "and", "asm", "auto", "bool", "break", "case", "catch", "char", "class", "concept", "const", "consteval", "constexpr", "constinit", "continue", "decltype", "default", "delete", "do", "double", "else", "enum", "explicit", "export", "extern", "false", "float", "for", "friend", "goto", "if", "inline", "int", "long", "mutable", "namespace", "new", "noexcept", "not", "nullptr", "operator", "or", "private", "protected", "public", "register", "requires", "return", "short", "signed", "sizeof", "static", "struct", "switch", "template", "this", "throw", "true", "try", "typedef", "typeid", "typename", "union", "unsigned", "using", "virtual", "void", "volatile", "while", "xor"};

constexpr std::string_view kStandardHeaders = "algorithm any array atomic bit bitset cassert cctype cfloat charconv chrono cinttypes climits cmath compare complex concepts cstddef cstdint cstdio cstdlib cstring deque exception expected filesystem format forward_list functional initializer_list iomanip ios iosfwd iostream istream iterator limits list locale map memory memory_resource mutex new numbers numeric optional ostream print queue random ranges ratio regex set shared_mutex source_location span sstream stack stdexcept stop_token streambuf string string_view system_error thread tuple type_traits typeindex typeinfo unordered_map unordered_set utility valarray variant vector";

struct Violation {
    std::string where; // file:line:column, relative to its root
    std::string kind;
    std::string name;
    std::string rule;
    std::string expected;
    std::string proposal;     // the spelling the rule asks for, before it is checked against its scope
    std::string scope;        // the USR of the scope the name is declared in, where a proposal must not clash
    std::string proposalForm; // the rule's form the proposal must match: UpperCamel, lowerCamel, snake, …
};

struct Rule {
    std::string_view id;
    std::string_view description;
};

constexpr std::array<Rule, 15> kRules{{
    {"R1", "types, enums and aliases: UpperCamel; std-style traits `*_t`, `is_*`, `has_*`, a member alias `*_type`"},
    {"R2", "concepts: UpperCamel or std-style snake_case"},
    {"R3", "enumerators: UPPER_SNAKE, UpperCamel or lowerCamel, one case per enum (the minority is reported)"},
    {"R4", "namespaces: lowercase"},
    {"R5", "functions and methods: lowerCamel; a non-public method may start with `_`"},
    {"R6", "a Block's reflected settings: snake_case; a reflected `_` field in either case"},
    {"R7", "other public fields and ports: lowerCamel; a public `_` field in either case"},
    {"R8", "private and protected fields: `_`, then either case"},
    {"R9", "locals and parameters: lowerCamel; mathematical notation (`N`, `x_0`, `S_k`); a trailing `_` for a constructor parameter, a keyword or a member's name; Boost.UT suites any name"},
    {"R10", "constants: kUpperCamel for every fixed value at any scope (constexpr, constinit, a constant-type const at namespace scope); not a lambda, not a value computed from template parameters"},
    {"R11", "type template parameters: UpperCamel (`T`, `TBlock`)"},
    {"R12", "non-type template parameters: lowerCamel, UpperCamel or UPPER_SNAKE"},
    {"R13", "macros: UPPER_SNAKE; a lowercase macro that mirrors a keyword or attribute (`forceinline`) is accepted"},
    {"R14", "files: a header shares a word with what it declares; a .cpp is a header's stem with a suffix; a test is `qa_`/`bm_` + a header, type or function as spelled, or a concern"},
    {"R15", "directories: lowercase"},
}};

std::string_view ruleDescription(std::string_view id) {
    const auto rule = std::ranges::find(kRules, id, &Rule::id);
    return rule == kRules.end() ? std::string_view{} : rule->description;
}

int ruleNumber(std::string_view id) {
    int number = 0;
    std::from_chars(id.data() + 1, id.data() + id.size(), number);
    return number;
}

enum class EnumeratorCase { upperSnake, upperCamel, lowerCamel };
constexpr std::array<std::string_view, 3> kEnumeratorCaseForms{"UPPER_SNAKE", "UpperCamel", "lowerCamel"};
static_assert(kEnumeratorCaseForms.size() == static_cast<std::size_t>(EnumeratorCase::lowerCamel) + 1UZ);

struct Enumerator {
    EnumeratorCase written;
    std::string    name;
    unsigned       line;
};

struct Findings {
    std::map<std::string, Violation>                                    violations;   // key: where + name
    std::map<std::string, std::map<std::string, Enumerator>>            enumerators;  // enum (USR) -> where -> enumerator
    std::map<std::string, std::map<std::string, std::set<std::string>>> observations; // ambiguity -> what was seen -> places
    std::map<std::string, std::set<std::string>>                        checked;      // rule -> places checked
    std::map<std::string, std::set<std::string>>                        declaredIn;   // file -> names declared there
    std::map<std::string, std::set<std::string>>                        namesInScope; // scope (USR) -> names declared in it
    std::set<std::string>                                               unitsWithErrors;
};

struct Scope {
    std::vector<fs::path>    roots;
    std::vector<fs::path>    excluded;
    std::vector<std::string> stdStyleNamespaces; // `gr::meta`
    std::vector<fs::path>    stdStyleDirectories;

    static bool        under(const std::string& path, const fs::path& directory) { return path.starts_with(directory.string() + "/"); }
    [[nodiscard]] bool inside(const std::string& path) const {
        return std::ranges::any_of(roots, [&path](const fs::path& root) { return under(path, root); }) && std::ranges::none_of(excluded, [&path](const fs::path& directory) { return under(path, directory); });
    }
    [[nodiscard]] std::string relative(const std::string& path) const {
        for (const fs::path& root : roots) {
            if (under(path, root)) {
                return (root.filename() / fs::relative(path, root)).string();
            }
        }
        return path;
    }
};

/// the declaration's head, up to its initialiser, and the initialiser's tokens
struct Declaration {
    bool                     constexprOrConstinit = false;
    std::vector<std::string> initialiser;
};

Declaration declarationOf(CXTranslationUnit unit, CXCursor cursor) {
    CXToken* tokens     = nullptr;
    unsigned tokenCount = 0;
    clang_tokenize(unit, clang_getCursorExtent(cursor), &tokens, &tokenCount);
    Declaration declaration;
    bool        inInitialiser = false;
    for (unsigned i = 0; i < tokenCount; ++i) {
        const std::string spelling = taken(clang_getTokenSpelling(unit, tokens[i]));
        if (inInitialiser) {
            declaration.initialiser.push_back(spelling);
        } else if (spelling == "=" || spelling == "{") {
            inInitialiser = true;
        } else {
            declaration.constexprOrConstinit = declaration.constexprOrConstinit || spelling == "constexpr" || spelling == "constinit";
        }
    }
    clang_disposeTokens(unit, tokens, tokenCount);
    return declaration;
}

bool derivesFromBlock(CXCursor record, int inheritanceDepth = 0) {
    bool derives = false;
    if (inheritanceDepth > 8) {
        return false;
    }
    struct BaseSearch {
        bool* derives;
        int   inheritanceDepth;
    } baseSearch{&derives, inheritanceDepth};
    clang_visitChildren(
        record,
        [](CXCursor child, CXCursor, CXClientData searchContext) {
            auto& search = *static_cast<BaseSearch*>(searchContext);
            if (clang_getCursorKind(child) != CXCursor_CXXBaseSpecifier) {
                return CXChildVisit_Continue;
            }
            if (std::regex_search(taken(clang_getTypeSpelling(clang_getCursorType(child))), kBlockBase)) {
                *search.derives = true;
                return CXChildVisit_Break;
            }
            const CXCursor baseDefinition = clang_getCursorDefinition(clang_getTypeDeclaration(clang_getCursorType(child)));
            if (!clang_Cursor_isNull(baseDefinition) && derivesFromBlock(baseDefinition, search.inheritanceDepth + 1)) {
                *search.derives = true;
                return CXChildVisit_Break;
            }
            return CXChildVisit_Continue;
        },
        &baseSearch);
    return derives;
}

bool isRecord(CXCursorKind kind) { return kind == CXCursor_StructDecl || kind == CXCursor_ClassDecl || kind == CXCursor_UnionDecl || kind == CXCursor_ClassTemplate || kind == CXCursor_ClassTemplatePartialSpecialization; }

bool isFunctionLike(CXCursorKind kind) { return kind == CXCursor_FunctionDecl || kind == CXCursor_CXXMethod || kind == CXCursor_FunctionTemplate || kind == CXCursor_LambdaExpr || kind == CXCursor_Constructor || kind == CXCursor_Destructor || kind == CXCursor_ConversionFunction; }

// what a `const` (not `constexpr`) global holds when it is a constant: a number, an enum, a pointer, a string_view, or an array of these
bool isConstantType(CXType type) {
    const CXType canonicalType = clang_getCanonicalType(type);
    if ((canonicalType.kind >= CXType_FirstBuiltin && canonicalType.kind <= CXType_LastBuiltin) || canonicalType.kind == CXType_Enum || canonicalType.kind == CXType_Pointer) {
        return true;
    }
    if (canonicalType.kind == CXType_ConstantArray || canonicalType.kind == CXType_IncompleteArray) {
        return isConstantType(clang_getArrayElementType(canonicalType));
    }
    const std::string typeSpelling = taken(clang_getTypeSpelling(canonicalType));
    if (typeSpelling.contains("basic_string_view")) {
        return true;
    }
    if (typeSpelling.contains("std::") && typeSpelling.contains("array<")) { // a std::array is a constant when what it holds is
        return isConstantType(clang_Type_getTemplateArgumentAsType(canonicalType, 0));
    }
    return false;
}

bool isClosure(CXType type) { return taken(clang_getTypeSpelling(clang_getCanonicalType(type))).contains("lambda"); }

bool isInStd(CXCursor cursor) {
    for (CXCursor scope = clang_getCursorSemanticParent(cursor); (clang_Cursor_isNull(scope) == 0) && clang_getCursorKind(scope) != CXCursor_TranslationUnit; scope = clang_getCursorSemanticParent(scope)) {
        if (clang_getCursorKind(scope) == CXCursor_Namespace && taken(clang_getCursorSpelling(scope)) == "std") {
            return true;
        }
    }
    return false;
}

std::string qualifiedNamespaceOf(CXCursor cursor) {
    std::vector<std::string> names;
    for (CXCursor scope = clang_getCursorSemanticParent(cursor); (clang_Cursor_isNull(scope) == 0) && clang_getCursorKind(scope) != CXCursor_TranslationUnit; scope = clang_getCursorSemanticParent(scope)) {
        if (clang_getCursorKind(scope) == CXCursor_Namespace) {
            names.insert(names.begin(), taken(clang_getCursorSpelling(scope)));
        }
    }
    return joinedWith(names, "::");
}

std::set<std::string> templateParametersAround(CXCursor cursor) {
    std::set<std::string> names;
    for (CXCursor scope = clang_getCursorSemanticParent(cursor); (clang_Cursor_isNull(scope) == 0) && clang_getCursorKind(scope) != CXCursor_TranslationUnit; scope = clang_getCursorSemanticParent(scope)) {
        clang_visitChildren(
            scope,
            [](CXCursor child, CXCursor, CXClientData namesContext) {
                const CXCursorKind kind = clang_getCursorKind(child);
                if (kind == CXCursor_TemplateTypeParameter || kind == CXCursor_NonTypeTemplateParameter || kind == CXCursor_TemplateTemplateParameter) {
                    static_cast<std::set<std::string>*>(namesContext)->insert(taken(clang_getCursorSpelling(child)));
                }
                return CXChildVisit_Continue;
            },
            &names);
    }
    return names;
}

std::set<std::string> memberNamesOf(CXCursor record) {
    std::set<std::string> names;
    clang_visitChildren(
        record,
        [](CXCursor child, CXCursor, CXClientData namesContext) {
            const CXCursorKind kind = clang_getCursorKind(child);
            if (kind == CXCursor_FieldDecl || kind == CXCursor_CXXMethod || kind == CXCursor_VarDecl || kind == CXCursor_FunctionTemplate) {
                static_cast<std::set<std::string>*>(namesContext)->insert(taken(clang_getCursorSpelling(child)));
            }
            return CXChildVisit_Continue;
        },
        &names);
    return names;
}

// every name the standard library declares in `std` (members, aliases, concepts, functions, types), read from its headers with the build's own flags
std::optional<std::set<std::string>> standardNamesOf(std::string_view buildDirectory, const Scope& scope) {
    const fs::path source = fs::temp_directory_path() / std::format("naming-check-standard-{}.cpp", ::getpid());
    {
        std::ofstream scratchFile{source};
        for (const auto header : std::views::split(kStandardHeaders, ' ')) {
            scratchFile << std::format("#include <{}>\n", std::string_view{header});
        }
        if (!scratchFile) {
            std::println(stderr, "naming-check: cannot write {}", source.string());
            return std::nullopt;
        }
    }
    const std::vector<std::string> arguments = devtools::argumentsFor(buildDirectory, [&scope](const std::string& scratchFile) { return scope.inside(scratchFile); }, source.string());
    std::vector<const char*>       argumentPointers;
    std::ranges::transform(arguments, std::back_inserter(argumentPointers), &std::string::c_str);
    std::set<std::string> standardNames;
    CXIndex               clangIndex          = clang_createIndex(0, 0);
    CXTranslationUnit     standardHeadersUnit = nullptr;
    if (!argumentPointers.empty() && clang_parseTranslationUnit2FullArgv(clangIndex, nullptr, argumentPointers.data(), static_cast<int>(argumentPointers.size()), nullptr, 0, CXTranslationUnit_KeepGoing | CXTranslationUnit_SkipFunctionBodies, &standardHeadersUnit) == CXError_Success) {
        clang_visitChildren(
            clang_getTranslationUnitCursor(standardHeadersUnit),
            [](CXCursor cursor, CXCursor, CXClientData namesContext) {
                const std::string name = taken(clang_getCursorSpelling(cursor));
                if (clang_isDeclaration(clang_getCursorKind(cursor)) != 0 && !name.empty() && !name.starts_with('_') && !name.starts_with("operator") && !name.starts_with("(") && isInStd(cursor)) {
                    static_cast<std::set<std::string>*>(namesContext)->insert(name);
                }
                return clang_getCursorKind(cursor) == CXCursor_Namespace || isRecord(clang_getCursorKind(cursor)) || clang_getCursorKind(cursor) == CXCursor_LinkageSpec ? CXChildVisit_Recurse : CXChildVisit_Continue;
            },
            &standardNames);
        clang_disposeTranslationUnit(standardHeadersUnit);
    }
    clang_disposeIndex(clangIndex);
    fs::remove(source);
    return standardNames;
}

bool isStdSpecialisation(CXCursor record) {
    const CXCursor specialisedTemplate = clang_getSpecializedCursorTemplate(record);
    return (clang_Cursor_isNull(specialisedTemplate) == 0) && isInStd(specialisedTemplate);
}

bool overrides(CXCursor method) {
    CXCursor* overridden    = nullptr;
    unsigned  overrideCount = 0;
    clang_getOverriddenCursors(method, &overridden, &overrideCount);
    clang_disposeOverriddenCursors(overridden);
    return overrideCount > 0;
}

struct Walk {
    CXTranslationUnit                                   translationUnit;
    const Scope*                                        scope;
    Findings*                                           findings;
    const std::map<std::string, std::set<std::string>>* reflected; // record name -> members named in GR_MAKE_REFLECTABLE
    const std::set<std::string>*                        standardNames;
    std::unordered_map<std::string, std::string>        canonicalPaths; // a unit names the same files over and over
};

void observe(Walk& walk, const std::string& ambiguity, const std::string& what, const std::string& where) { walk.findings->observations[ambiguity][what].insert(where); }

/// `std` names are what a member, an alias or a concept may borrow (`value_type`, `begin`, `push_back`); any other name follows its rule
enum class Borrows : bool { nothing, standardNames };

struct Checked {
    std::string kind;
    std::string rule;
    std::string expected;
    std::string proposalForm;
};

[[nodiscard]] Checked lowerCamelOrStd(std::string kind, std::string rule, bool stdStyle) { return Checked{.kind = std::move(kind), .rule = std::move(rule), .expected = stdStyle ? "snake_case (std-style)" : "lowerCamel", .proposalForm = stdStyle ? "snake" : "lowerCamel"}; }

void report(Walk& walk, CXCursor cursor, const std::string& where, const std::string& name, const Checked& checked) { walk.findings->violations.try_emplace(where + " " + name, Violation{.where = where, .kind = checked.kind, .name = name, .rule = checked.rule, .expected = checked.expected, .proposal = spellingFor(name, checked.proposalForm), .scope = taken(clang_getCursorUSR(clang_getCursorSemanticParent(cursor))), .proposalForm = checked.proposalForm}); }

void check(Walk& walk, CXCursor cursor, const std::string& where, const std::string& name, const Checked& checked, const std::regex& pattern, Borrows borrows = Borrows::nothing) {
    walk.findings->checked[checked.rule].insert(where);
    if (std::regex_match(name, pattern)) {
        return;
    }
    if (borrows == Borrows::standardNames && walk.standardNames->contains(name)) {
        observe(walk, "A6", "standard-library name of a member, alias or concept", where + " " + name);
        return;
    }
    report(walk, cursor, where, name, checked);
}

bool isStdStyle(const Walk& walk, CXCursor cursor, const std::string& file) {
    const std::string space = qualifiedNamespaceOf(cursor);
    return std::ranges::any_of(walk.scope->stdStyleNamespaces, [&space](const std::string& stdStyle) { return space == stdStyle || space.starts_with(stdStyle + "::"); }) || std::ranges::any_of(walk.scope->stdStyleDirectories, [&file](const fs::path& directory) { return Scope::under(file, directory); });
}

const std::string& canonicalOf(Walk& walk, CXFile file) {
    const std::string spelled = taken(clang_getFileName(file));
    auto              known   = walk.canonicalPaths.find(spelled);
    if (known == walk.canonicalPaths.end()) {
        known = walk.canonicalPaths.emplace(spelled, fs::weakly_canonical(spelled).string()).first;
    }
    return known->second;
}

void checkEnumerator(Walk& walk, CXCursor cursor, const std::string& where, const std::string& name, unsigned line, bool stdStyle) {
    walk.findings->checked["R3"].insert(where);
    if (std::regex_match(name, kSingleLetter)) {
        return; // a single letter is written in every case: it counts for none
    }
    const bool lower  = std::regex_match(name, kLowerCamel) || (stdStyle && std::regex_match(name, kSnake));
    const auto caseOf = [&]() -> std::optional<EnumeratorCase> {
        if (std::regex_match(name, kUpperSnake)) {
            return EnumeratorCase::upperSnake;
        }
        if (std::regex_match(name, kUpperCamel)) {
            return EnumeratorCase::upperCamel;
        }
        return lower ? std::optional{EnumeratorCase::lowerCamel} : std::nullopt;
    };
    const std::optional<EnumeratorCase> written = caseOf();
    if (!written) {
        report(walk, cursor, where, name, Checked{.kind = "enumerator", .rule = "R3", .expected = "UPPER_SNAKE, UpperCamel or lowerCamel", .proposalForm = "lowerCamel"});
        return;
    }
    walk.findings->enumerators[taken(clang_getCursorUSR(clang_getCursorSemanticParent(cursor)))].try_emplace(where, Enumerator{.written = *written, .name = name, .line = line});
}

void checkParameterOrLocal(Walk& walk, CXCursor cursor, const std::string& where, const std::string& name, const std::string& kind, bool stdStyle) {
    const std::string stem = name.ends_with('_') ? name.substr(0, name.size() - 1U) : name;
    if (name == "_" || std::regex_match(name, kSingleLetter) || std::regex_match(name, kMathematical)) {
        walk.findings->checked["R9"].insert(where);
        return;
    }
    if (name.ends_with('_') && !stem.empty()) {
        const CXCursor function      = clang_getCursorSemanticParent(cursor);
        const CXCursor record        = clang_getCursorSemanticParent(function);
        const bool     isConstructor = clang_getCursorKind(function) == CXCursor_Constructor;
        const bool     avoidsMember  = isRecord(clang_getCursorKind(record)) && memberNamesOf(record).contains(stem);
        const bool     dodgesKeyword = kKeywords.contains(stem);
        if (isConstructor || avoidsMember || dodgesKeyword) {
            check(walk, cursor, where, stem, Checked{.kind = kind, .rule = "R9", .expected = "lowerCamel", .proposalForm = "lowerCamel"}, stdStyle ? kSnake : kLowerCamel);
            return;
        }
    }
    check(walk, cursor, where, name, Checked{.kind = kind, .rule = "R9", .expected = stdStyle ? "snake_case" : "lowerCamel", .proposalForm = stdStyle ? "snake" : "lowerCamel"}, stdStyle && std::regex_match(name, kSnake) ? kSnake : kLowerCamel);
}

/// `auto [a, b] = …`: libclang leaves a structured binding unexposed, so its names are read from its tokens
void checkStructuredBinding(Walk& walk, CXCursor cursor, const std::string& relativeFile, bool stdStyle) {
    CXToken* tokens     = nullptr;
    unsigned tokenCount = 0;
    clang_tokenize(walk.translationUnit, clang_getCursorExtent(cursor), &tokens, &tokenCount);
    bool inBrackets = false;
    for (unsigned i = 0; i < tokenCount; ++i) {
        const std::string spelling = taken(clang_getTokenSpelling(walk.translationUnit, tokens[i]));
        if (spelling == "=" || spelling == ":" || (spelling == "]" && inBrackets)) {
            break;
        }
        if (spelling == "[") {
            inBrackets = true;
        } else if (inBrackets && clang_getTokenKind(tokens[i]) == CXToken_Identifier) {
            unsigned line   = 0;
            unsigned column = 0;
            clang_getExpansionLocation(clang_getTokenLocation(walk.translationUnit, tokens[i]), nullptr, &line, &column, nullptr);
            checkParameterOrLocal(walk, cursor, std::format("{}:{}:{}", relativeFile, line, column), spelling, "structured binding", stdStyle);
        }
    }
    clang_disposeTokens(walk.translationUnit, tokens, tokenCount);
}

void checkField(Walk& walk, CXCursor cursor, const std::string& where, const std::string& name, bool stdStyle) {
    const CXCursor record            = clang_getCursorSemanticParent(cursor);
    const auto     access            = clang_getCXXAccessSpecifier(cursor);
    const auto     reflectedOfRecord = walk.reflected->find(taken(clang_getCursorSpelling(record)));
    const bool     isReflected       = reflectedOfRecord != walk.reflected->end() && reflectedOfRecord->second.contains(name);
    const bool     isPort            = std::regex_search(taken(clang_getTypeSpelling(clang_getCursorType(cursor))), kPortType);
    if (access == CX_CXXPrivate || access == CX_CXXProtected) {
        check(walk, cursor, where, name, Checked{.kind = "private field", .rule = "R8", .expected = "_ then either case", .proposalForm = "_lowerCamel"}, kUnderscored);
    } else if (isPort) {
        check(walk, cursor, where, name, Checked{.kind = "port", .rule = "R7", .expected = "lowerCamel", .proposalForm = "lowerCamel"}, kLowerCamel); // reflected, but not a setting
    } else if (name.starts_with('_')) {
        check(walk, cursor, where, name, Checked{.kind = isReflected ? "reflected `_` field" : "public `_` field", .rule = isReflected ? "R6" : "R7", .expected = "_ then either case", .proposalForm = "_lowerCamel"}, kUnderscored);
    } else if (isReflected && derivesFromBlock(record)) {
        check(walk, cursor, where, name, Checked{.kind = "reflected setting", .rule = "R6", .expected = "snake_case", .proposalForm = "snake"}, kSnake);
    } else {
        check(walk, cursor, where, name, lowerCamelOrStd("public field", "R7", stdStyle), isReflected ? kSnakeOrLowerCamel : stdStyle ? kSnake : kLowerCamel, Borrows::standardNames);
    }
}

void checkVariable(Walk& walk, CXCursor cursor, const std::string& where, const std::string& name, CXCursorKind parentKind, bool stdStyle) {
    const bool        isLocal     = isFunctionLike(parentKind);
    const CXType      type        = clang_getCursorType(cursor);
    const Declaration declaration = declarationOf(walk.translationUnit, cursor);
    if (std::regex_search(taken(clang_getTypeSpelling(type)), kBoostSuite)) {
        observe(walk, "R9", "Boost.UT suite object (any name)", where + " " + name);
        return;
    }
    const bool isConstant = !isClosure(type) && (declaration.constexprOrConstinit || (!isLocal && clang_isConstQualifiedType(type) != 0 && isConstantType(type)));
    if (isConstant) {
        const std::set<std::string> parameters  = templateParametersAround(cursor);
        const bool                  isDependent = std::ranges::any_of(declaration.initialiser, [&parameters](const std::string& token) { return parameters.contains(token); });
        if (isDependent && (std::regex_match(name, kConstexprName) || std::regex_match(name, kLowerCamel) || std::regex_match(name, kSnake))) {
            observe(walk, "R10", "computed from template parameters (either form)", where + " " + name);
            return;
        }
        check(walk, cursor, where, name, Checked{.kind = isLocal ? "local constant" : isRecord(parentKind) ? "static constant" : "constant", .rule = "R10", .expected = "kConstant", .proposalForm = "kConstant"}, stdStyle && std::regex_match(name, kSnake) ? kSnake : kConstexprName);
        return;
    }
    if (isRecord(parentKind)) {
        const auto access = clang_getCXXAccessSpecifier(cursor);
        if (access == CX_CXXPrivate || access == CX_CXXProtected) {
            check(walk, cursor, where, name, Checked{.kind = "private static field", .rule = "R8", .expected = "_ then either case", .proposalForm = "_lowerCamel"}, kUnderscored);
        } else {
            check(walk, cursor, where, name, lowerCamelOrStd("public static field", "R7", stdStyle), name.starts_with('_') ? kUnderscored : stdStyle ? kSnake : kLowerCamel, Borrows::standardNames);
        }
        return;
    }
    checkParameterOrLocal(walk, cursor, where, name, isLocal ? "local" : "variable", stdStyle);
}

CXChildVisitResult visitDeclaration(CXCursor cursor, CXCursor, CXClientData walkContext) {
    Walk&    walk          = *static_cast<Walk*>(walkContext);
    unsigned line          = 0;
    unsigned column        = 0;
    CXFile   expansionFile = nullptr;
    clang_getExpansionLocation(clang_getCursorLocation(cursor), &expansionFile, &line, &column, nullptr);
    const std::string  file         = expansionFile == nullptr ? std::string{} : canonicalOf(walk, expansionFile);
    const CXCursorKind kind         = clang_getCursorKind(cursor);
    CXFile             spellingFile = nullptr;
    unsigned           spellingLine = 0;
    clang_getSpellingLocation(clang_getCursorLocation(cursor), &spellingFile, &spellingLine, nullptr, nullptr);
    if (!walk.scope->inside(file)) {
        return kind == CXCursor_Namespace || kind == CXCursor_LinkageSpec || kind == CXCursor_UnexposedDecl ? CXChildVisit_Recurse : CXChildVisit_Continue;
    }
    const std::string name     = taken(clang_getCursorSpelling(cursor));
    const std::string where    = std::format("{}:{}:{}", walk.scope->relative(file), line, column);
    const bool        stdStyle = isStdStyle(walk, cursor, file);
    if (expansionFile != nullptr && spellingFile != nullptr && (clang_File_isEqual(expansionFile, spellingFile) == 0 || spellingLine != line)) {
        observe(walk, "R13", "declaration a macro writes (its macro's name is checked, not this)", where + " " + name);
        return CXChildVisit_Continue;
    }
    if (clang_isDeclaration(kind) != 0 && !name.empty()) {
        walk.findings->declaredIn[file].insert(name.substr(0, name.find('<')));
        walk.findings->namesInScope[taken(clang_getCursorUSR(clang_getCursorSemanticParent(cursor)))].insert(name);
    }
    const CXCursorKind parentKind = clang_getCursorKind(clang_getCursorSemanticParent(cursor));
    if (kind == CXCursor_UnexposedDecl && isFunctionLike(parentKind)) {
        checkStructuredBinding(walk, cursor, walk.scope->relative(file), stdStyle);
        return CXChildVisit_Recurse;
    }
    if (clang_isCursorDefinition(cursor) == 0 && kind != CXCursor_ParmDecl && kind != CXCursor_FieldDecl && kind != CXCursor_EnumConstantDecl && kind != CXCursor_TemplateTypeParameter && kind != CXCursor_NonTypeTemplateParameter && kind != CXCursor_TemplateTemplateParameter && kind != CXCursor_Namespace && kind != CXCursor_TypeAliasDecl && kind != CXCursor_TypedefDecl && kind != CXCursor_ConceptDecl && kind != CXCursor_VarDecl) {
        return CXChildVisit_Recurse; // declarations are checked where they are defined
    }
    if (name.empty() || clang_Cursor_isAnonymous(cursor) != 0 || name.starts_with("(") || name.starts_with("operator") || name == "main") {
        return CXChildVisit_Recurse;
    }

    switch (kind) {
    case CXCursor_StructDecl:
    case CXCursor_ClassDecl:
    case CXCursor_UnionDecl:
    case CXCursor_ClassTemplate:
    case CXCursor_EnumDecl:
    case CXCursor_TypedefDecl:
    case CXCursor_TypeAliasDecl:
    case CXCursor_TypeAliasTemplateDecl: {
        if (isRecord(kind) && isStdSpecialisation(cursor)) {
            return CXChildVisit_Continue; // names inside a std:: specialisation are the standard's
        }
        if (kind == CXCursor_TypeAliasDecl && parentKind == CXCursor_TypeAliasTemplateDecl) {
            return CXChildVisit_Continue; // checked as the alias template
        }
        if (std::regex_search(name, kAcronymRun)) {
            observe(walk, "A4", "acronym in capitals (type)", where + " " + name);
        }
        const bool isAlias = kind == CXCursor_TypedefDecl || kind == CXCursor_TypeAliasDecl || kind == CXCursor_TypeAliasTemplateDecl;
        if (isAlias && isRecord(parentKind) && std::regex_match(name, kMemberTypeName)) {
            walk.findings->checked["R1"].insert(where);
            return CXChildVisit_Continue;
        }
        check(walk, cursor, where, name, Checked{.kind = "type", .rule = "R1", .expected = stdStyle ? "snake_case (std-style)" : "UpperCamel (or a std-style trait)", .proposalForm = stdStyle ? "snake" : "UpperCamel"}, stdStyle ? (isAlias ? kStdStyleAliasName : kSnake) : kTypeName, isAlias ? Borrows::standardNames : Borrows::nothing);
        return kind == CXCursor_TypeAliasTemplateDecl ? CXChildVisit_Continue : CXChildVisit_Recurse;
    }
    case CXCursor_ConceptDecl: check(walk, cursor, where, name, Checked{.kind = "concept", .rule = "R2", .expected = "UpperCamel or std-style snake_case", .proposalForm = "UpperCamel"}, kConceptName, Borrows::standardNames); return CXChildVisit_Recurse; // its template parameters are checked too
    case CXCursor_EnumConstantDecl: checkEnumerator(walk, cursor, where, name, line, stdStyle); return CXChildVisit_Continue;
    case CXCursor_Namespace: check(walk, cursor, where, name, Checked{.kind = "namespace", .rule = "R4", .expected = "lowercase", .proposalForm = "snake"}, kSnake); return CXChildVisit_Recurse;
    case CXCursor_FunctionDecl:
    case CXCursor_CXXMethod:
    case CXCursor_FunctionTemplate: {
        const bool isOverride = overrides(cursor);
        if (clang_getCursorKind(clang_getCursorLexicalParent(cursor)) == CXCursor_LinkageSpec || isOverride) {
            observe(walk, "A6", isOverride ? "override (skipped)" : "extern C (skipped)", where + " " + name);
            return CXChildVisit_Recurse;
        }
        if (isRecord(parentKind) && isStdSpecialisation(clang_getCursorSemanticParent(cursor))) {
            return CXChildVisit_Continue;
        }
        const std::string recordName = taken(clang_getCursorSpelling(clang_getCursorSemanticParent(cursor)));
        if (name.starts_with("<deduction guide") || (!recordName.empty() && name.substr(0, name.find('<')) == recordName.substr(0, recordName.find('<')))) {
            return CXChildVisit_Recurse; // a constructor template or deduction guide carries the type's name
        }
        const bool isMember  = isRecord(parentKind);
        const auto access    = clang_getCXXAccessSpecifier(cursor);
        const bool nonPublic = isMember && (access == CX_CXXPrivate || access == CX_CXXProtected);
        if (nonPublic && std::regex_match(name, kUnderscored)) {
            walk.findings->checked["R5"].insert(where);
            return CXChildVisit_Recurse; // a non-public method may start with `_`, in either case
        }
        check(walk, cursor, where, name, lowerCamelOrStd("function", "R5", stdStyle), stdStyle ? kSnake : kLowerCamel, isMember ? Borrows::standardNames : Borrows::nothing);
        return CXChildVisit_Recurse;
    }
    case CXCursor_FieldDecl: checkField(walk, cursor, where, name, stdStyle); return CXChildVisit_Continue;
    case CXCursor_VarDecl:
        if (isRecord(clang_getCursorKind(clang_getCursorSemanticParent(cursor))) && !isRecord(clang_getCursorKind(clang_getCursorLexicalParent(cursor)))) {
            return CXChildVisit_Recurse; // a static member defined out of its class: checked where it is declared
        }
        checkVariable(walk, cursor, where, name, parentKind, stdStyle);
        return CXChildVisit_Recurse;
    case CXCursor_ParmDecl: checkParameterOrLocal(walk, cursor, where, name, "parameter", stdStyle); return CXChildVisit_Continue;
    case CXCursor_TemplateTypeParameter:
    case CXCursor_TemplateTemplateParameter: check(walk, cursor, where, name, Checked{.kind = "type template parameter", .rule = "R11", .expected = "UpperCamel", .proposalForm = "UpperCamel"}, kUpperCamel); return CXChildVisit_Continue;
    case CXCursor_NonTypeTemplateParameter: check(walk, cursor, where, name, Checked{.kind = "non-type template parameter", .rule = "R12", .expected = "lowerCamel, UpperCamel or UPPER_SNAKE", .proposalForm = "lowerCamel"}, kNonTypeParameter); return CXChildVisit_Continue;
    default: return CXChildVisit_Recurse;
    }
}

std::map<std::string, std::set<std::string>> reflectedMembers(CXTranslationUnit unit) {
    std::map<std::string, std::set<std::string>> reflected;
    clang_visitChildren(
        clang_getTranslationUnitCursor(unit),
        [](CXCursor cursor, CXCursor, CXClientData reflectedContext) {
            if (clang_getCursorKind(cursor) != CXCursor_MacroExpansion || taken(clang_getCursorSpelling(cursor)) != "GR_MAKE_REFLECTABLE") {
                return CXChildVisit_Continue;
            }
            auto&                    members       = *static_cast<std::map<std::string, std::set<std::string>>*>(reflectedContext);
            CXTranslationUnit        expansionUnit = clang_Cursor_getTranslationUnit(cursor);
            CXToken*                 tokens        = nullptr;
            unsigned                 tokenCount    = 0;
            std::vector<std::string> macroArguments;
            clang_tokenize(expansionUnit, clang_getCursorExtent(cursor), &tokens, &tokenCount);
            int bracketDepth = 0;
            for (unsigned i = 0; i < tokenCount; ++i) {
                const std::string spelling = taken(clang_getTokenSpelling(expansionUnit, tokens[i]));
                bracketDepth += spelling == "(" || spelling == "<" ? 1 : spelling == ")" || spelling == ">" ? -1 : 0;
                if (bracketDepth == 1 && clang_getTokenKind(tokens[i]) == CXToken_Identifier) {
                    macroArguments.push_back(spelling);
                }
            }
            clang_disposeTokens(expansionUnit, tokens, tokenCount);
            if (!macroArguments.empty()) {
                members[macroArguments.front()].insert(macroArguments.begin() + 1, macroArguments.end());
            }
            return CXChildVisit_Continue;
        },
        &reflected);
    return reflected;
}

void checkMacros(CXTranslationUnit unit, const Scope& scope, Findings& findings) {
    struct MacroCheck {
        const Scope* scope;
        Findings*    findings;
    } macroCheck{&scope, &findings};
    clang_visitChildren(
        clang_getTranslationUnitCursor(unit),
        [](CXCursor cursor, CXCursor, CXClientData checkContext) {
            auto& macros = *static_cast<MacroCheck*>(checkContext);
            if (clang_getCursorKind(cursor) != CXCursor_MacroDefinition || clang_Cursor_isMacroBuiltin(cursor) != 0) {
                return CXChildVisit_Continue;
            }
            unsigned          line = 0;
            const std::string file = fileOf(clang_getCursorLocation(cursor), &line);
            if (!macros.scope->inside(file)) {
                return CXChildVisit_Continue;
            }
            const std::string name  = taken(clang_getCursorSpelling(cursor));
            const std::string where = std::format("{}:{}", macros.scope->relative(file), line);
            macros.findings->checked["R13"].insert(where);
            // the whole word first: `noreturn` is an attribute, not `no` + `return`
            const std::size_t prefixLength   = name.starts_with("force") ? 5U : name.starts_with("always_") ? 7U : name.starts_with("no") ? 2U : 0U;
            const bool        mirrorsKeyword = kKeywordLikeWords.contains(name) || kKeywords.contains(name) || kKeywordLikeWords.contains(std::string_view{name}.substr(prefixLength));
            if (mirrorsKeyword) {
                macros.findings->observations["A13"]["keyword-like macro (accepted)"].insert(name);
            } else if (!std::regex_match(name, kUpperSnake)) {
                macros.findings->violations.try_emplace(where + " " + name, Violation{.where = where, .kind = "macro", .name = name, .rule = "R13", .expected = "UPPER_SNAKE", .proposal = spellingFor(name, "UPPER_SNAKE"), .scope = {}, .proposalForm = "UPPER_SNAKE"});
            }
            return CXChildVisit_Continue;
        },
        &macroCheck);
}

std::vector<fs::path> sourceFiles(const Scope& scope) {
    std::vector<fs::path> files;
    for (const fs::path& root : scope.roots) {
        for (const auto& directoryEntry : fs::recursive_directory_iterator(root)) {
            const std::string path = fs::weakly_canonical(directoryEntry.path()).string();
            if (scope.inside(path) && (directoryEntry.is_directory() || directoryEntry.path().extension() == ".hpp" || directoryEntry.path().extension() == ".cpp")) {
                files.emplace_back(path);
            }
        }
    }
    return files;
}

// a header shares a word with something it declares; a .cpp is a header's stem (and a suffix); `qa_`/`bm_` + a header,
// type or function of the project as spelled, or a concern in UpperCamel; directories lowercase
void checkFiles(const Scope& scope, Findings& findings) {
    const std::vector<fs::path> files = sourceFiles(scope);
    std::set<std::string>       headerStems;
    std::set<std::string>       projectNames;
    for (const fs::path& path : files) {
        if (path.extension() == ".hpp") {
            headerStems.insert(path.stem().string());
        }
    }
    for (const auto& [file, names] : findings.declaredIn) {
        projectNames.insert(names.begin(), names.end());
    }
    for (const fs::path& path : files) {
        const std::string where = scope.relative(path.string());
        const std::string name  = path.filename().string();
        if (fs::is_directory(path)) {
            findings.checked["R15"].insert(where);
            if (!std::regex_match(name, kDirectoryName)) {
                findings.violations.try_emplace(where, Violation{.where = where, .kind = "directory", .name = name, .rule = "R15", .expected = "lowercase", .proposal = {}, .scope = {}, .proposalForm = {}});
            }
            continue;
        }
        const std::string stem         = path.stem().string();
        const auto        declarations = findings.declaredIn.find(path.string());
        if (declarations == findings.declaredIn.end()) {
            findings.observations["A12"]["file no unit includes (not checked)"].insert(where);
            continue;
        }
        findings.checked["R14"].insert(where);
        const bool        isTestFile = stem.starts_with("qa_") || stem.starts_with("bm_");
        const std::string subject    = isTestFile ? stem.substr(3U) : stem;
        bool              isNamed    = false;
        if (isTestFile) {
            isNamed = headerStems.contains(subject) || projectNames.contains(subject) || std::regex_match(subject, kUpperCamel);
        } else if (path.extension() == ".cpp") {
            isNamed = stem == "main" || where.starts_with("devtools/") || std::ranges::any_of(headerStems, [&stem](const std::string& header) { return stem.starts_with(header); });
        } else {
            const std::vector<std::string> stemWords = wordsOf(stem);
            isNamed                                  = std::ranges::any_of(declarations->second, [&stemWords](const std::string& declared) { return std::ranges::any_of(wordsOf(declared), [&stemWords](const std::string& word) { return std::ranges::any_of(stemWords, [&word](const std::string& stemWord) { return stemWord == word || stemWord == word + "s" || stemWord == word + "es"; }); }); });
        }
        if (!isNamed) {
            const std::string expected = isTestFile ? "qa_/bm_ + a header, type or function of the project" : path.extension() == ".cpp" ? "a header's stem, with a suffix" : "a word of something it declares";
            findings.violations.try_emplace(where, Violation{.where = where, .kind = "file", .name = name, .rule = "R14", .expected = expected, .proposal = {}, .scope = {}, .proposalForm = {}});
        }
    }
}

/// headers that include nothing but the standard library and other such headers
std::vector<std::string> stdStyleCandidates(const Scope& scope) {
    const std::regex                                includeLine{R"(^\s*#\s*include\s*([<"])([^>"]+)[>"])"};
    std::map<std::string, std::vector<std::string>> projectIncludes; // header filename -> project headers it includes
    std::set<std::string>                           excluded;
    for (const fs::path& path : sourceFiles(scope)) {
        if (path.extension() != ".hpp") {
            continue;
        }
        std::ifstream header{path};
        std::string   line;
        auto&         includes = projectIncludes[path.filename().string()];
        while (std::getline(header, line)) {
            std::smatch include;
            if (!std::regex_search(line, include, includeLine)) {
                continue;
            }
            const std::string included = include[2].str();
            const bool        standard = include[1].str() == "<" && std::ranges::contains(std::views::split(kStandardHeaders, ' ') | std::views::transform([](auto word) { return std::string_view{word}; }), std::string_view{included});
            if (include[1].str() == "\"") {
                includes.push_back(fs::path{included}.filename().string());
            } else if (!standard) {
                excluded.insert(path.filename().string());
            }
        }
    }
    for (bool changed = true; changed;) { // a header that includes a non-candidate is no candidate either
        changed = false;
        for (const auto& [header, includes] : projectIncludes) {
            if (!excluded.contains(header) && std::ranges::any_of(includes, [&](const std::string& included) { return excluded.contains(included) || !projectIncludes.contains(included); })) {
                changed = excluded.insert(header).second || changed;
            }
        }
    }
    std::vector<std::string> candidates;
    for (const auto& [header, includes] : projectIncludes) {
        if (!excluded.contains(header)) {
            candidates.push_back(header);
        }
    }
    return candidates;
}

/// a proposal is made only when it is valid for its rule and nothing in its scope already has that name
void settleProposals(Findings& findings) {
    for (auto& [key, violation] : findings.violations) {
        const auto scopeNames = findings.namesInScope.find(violation.scope);
        const bool clashes    = scopeNames != findings.namesInScope.end() && scopeNames->second.contains(violation.proposal);
        const bool valid      = !violation.proposal.empty() && std::regex_match(violation.proposal, patternOf(violation.proposalForm));
        if (clashes || !valid || violation.proposal == violation.name) {
            violation.proposal.clear();
        }
    }
}

/// within one enum one case: the minority is reported, in the case of the majority (a tie keeps the first enumerator's)
void checkEnumeratorCases(Findings& findings) {
    for (const auto& [enumeration, enumerators] : findings.enumerators) {
        std::map<EnumeratorCase, std::size_t> counts;
        for (const auto& [where, enumerator] : enumerators) {
            ++counts[enumerator.written];
        }
        if (counts.size() < 2U) {
            continue;
        }
        const Enumerator& first    = *std::ranges::min_element(enumerators | std::views::values, {}, &Enumerator::line);
        EnumeratorCase    majority = first.written;
        for (const auto& [written, count] : counts) {
            majority = count > counts[majority] ? written : majority;
        }
        const std::string form{kEnumeratorCaseForms[static_cast<std::size_t>(majority)]};
        for (const auto& [where, enumerator] : enumerators) {
            if (enumerator.written != majority) {
                findings.violations.try_emplace(where + " " + enumerator.name, Violation{.where = where, .kind = "enumerator", .name = enumerator.name, .rule = "R3", .expected = form + ", as the rest of its enum", .proposal = spellingFor(enumerator.name, form), .scope = enumeration, .proposalForm = form});
            }
        }
    }
}

bool hasErrors(CXTranslationUnit unit, std::string& firstError) {
    for (unsigned index = 0; index < clang_getNumDiagnostics(unit); ++index) {
        CXDiagnostic diagnostic = clang_getDiagnostic(unit, index);
        const bool   isError    = clang_getDiagnosticSeverity(diagnostic) >= CXDiagnostic_Error;
        if (isError) {
            firstError = taken(clang_formatDiagnostic(diagnostic, clang_defaultDiagnosticDisplayOptions()));
        }
        clang_disposeDiagnostic(diagnostic);
        if (isError) {
            return true;
        }
    }
    return false;
}

std::optional<std::string> outputOf(const std::string& command) {
    std::FILE* pipe = ::popen(command.c_str(), "r");
    if (pipe == nullptr) {
        return std::nullopt;
    }
    std::string            output;
    std::array<char, 4096> buffer{};
    for (std::size_t count = 0; (count = std::fread(buffer.data(), 1, buffer.size(), pipe)) > 0;) {
        output.append(buffer.data(), count);
    }
    return ::pclose(pipe) == 0 ? std::optional{output} : std::nullopt;
}

/// the lines each file gained since `reference` (`git diff`, working tree included), by absolute path; line 0 marks a
/// file that is new or renamed, whose name is then checked too; nothing when git fails
std::optional<std::map<std::string, std::set<unsigned>>> changedLinesSince(const std::string& reference, const fs::path& inside) {
    if (reference.contains('\'') || reference.starts_with('-')) {
        return std::nullopt;
    }
    const std::optional<std::string> top = outputOf(std::format("git -C '{}' rev-parse --show-toplevel", inside.string()));
    if (!top) {
        return std::nullopt;
    }
    const fs::path                   repository{top->substr(0, top->find('\n'))};
    const std::optional<std::string> diff = outputOf(std::format("git -C '{}' diff --unified=0 --no-color --find-renames '{}' --", repository.string(), reference));
    if (!diff) {
        return std::nullopt;
    }
    std::map<std::string, std::set<unsigned>> changed;
    std::string                               file;
    bool                                      wholeFile = false;
    for (const auto lineRange : std::views::split(*diff, '\n')) {
        const std::string_view line{lineRange.begin(), lineRange.end()};
        if (line.starts_with("--- ")) {
            wholeFile = line == "--- /dev/null";
        } else if (line.starts_with("rename from ")) {
            wholeFile = true;
        } else if (line.starts_with("+++ b/")) {
            file = fs::weakly_canonical(repository / line.substr(6)).string();
            if (wholeFile) {
                changed[file].insert(0U);
            }
        } else if (line.starts_with("@@ ") && !file.empty()) {
            unsigned          first        = 0;
            unsigned          count        = 1;
            const std::size_t plus         = line.find('+');
            const char*       end          = line.data() + line.size();
            const auto [afterFirst, error] = std::from_chars(line.data() + plus + 1, end, first);
            if (error == std::errc{} && afterFirst < end && *afterFirst == ',') {
                std::from_chars(afterFirst + 1, end, count);
            }
            for (unsigned added = first; added < first + count; ++added) {
                changed[file].insert(added);
            }
        }
    }
    return changed;
}

/// line 0: a file or directory name
std::pair<std::string, unsigned> placeOf(const Violation& violation, const Scope& scope) {
    const std::size_t lineStart = violation.where.find(':');
    const std::string relative  = violation.where.substr(0, lineStart);
    unsigned          line      = 0;
    if (lineStart != std::string::npos) {
        std::from_chars(violation.where.data() + lineStart + 1, violation.where.data() + violation.where.size(), line);
    }
    for (const fs::path& root : scope.roots) {
        const std::string prefix = root.filename().string() + "/";
        if (relative.starts_with(prefix)) {
            return {(root / relative.substr(prefix.size())).string(), line};
        }
    }
    return {relative, line};
}

constexpr std::string_view kUsage = R"(usage: naming-check <clang-build-dir> --root <dir> [options]
       naming-check --rules | --help

  --root <dir>           check the sources under <dir> (repeatable)
  --exclude <dir>        leave out a directory under a root (repeatable)
  --std-style <ns|dir>   code in this namespace (gr::meta) or directory follows the standard library's naming (repeatable)
  --units <regex>        parse only the units whose path matches (repeatable; default: the units under the roots)
  --jobs <n>             parse on n threads (default 4)
  --rule <Rn>            report only these rules (repeatable)
  --changed <git-ref>    report only violations on lines changed since <git-ref>, the working tree included
  --list                 also list every violation, by rule, then file and line
  --summary              only the number of violations per rule, with the rule's name
  --table                only the violations, as a Markdown table (for a pull request or a commit message)
  --rules                list the rules and exit
  --help                 this text

exit: 0 no violations, 1 violations, 2 a usage error, a unit that could not be checked, or git failed)";

} // namespace

int main(int argc, char** argv) {
    const std::vector<std::string_view> arguments(argv + 1, argv + argc);
    Scope                               scope;
    std::vector<std::regex>             unitPatterns;
    std::set<std::string>               onlyRules;
    for (std::size_t i = 0; i + 1 < arguments.size(); ++i) {
        const std::string value{arguments[i + 1]};
        if (arguments[i] == "--root") {
            scope.roots.push_back(fs::weakly_canonical(value));
        } else if (arguments[i] == "--exclude") {
            scope.excluded.push_back(fs::weakly_canonical(value));
        } else if (arguments[i] == "--std-style") {
            if (fs::is_directory(value)) {
                scope.stdStyleDirectories.push_back(fs::weakly_canonical(value));
            } else {
                scope.stdStyleNamespaces.push_back(value);
            }
        } else if (arguments[i] == "--units") {
            unitPatterns.emplace_back(value);
        } else if (arguments[i] == "--rule") {
            onlyRules.insert(value);
        }
    }
    const auto has = [&arguments](std::string_view flag) { return std::ranges::contains(arguments, flag); };
    if (has("--help") || has("-h")) {
        std::println("{}", kUsage);
        return 0;
    }
    if (has("--rules")) {
        for (const Rule& rule : kRules) {
            std::println("{:<4} {}", rule.id, rule.description);
        }
        std::println("std-style code (--std-style) follows the standard library's naming, snake_case; std names stay where a standard concept expects them (value_type, push_back)");
        return 0;
    }
    const std::string_view jobsText    = devtools::valueAfter(arguments, "--jobs", "4");
    std::size_t            jobs        = 0;
    const bool             unknownRule = std::ranges::any_of(onlyRules, [](const std::string& rule) { return ruleDescription(rule).empty(); });
    if (arguments.empty() || arguments.front().starts_with("-") || scope.roots.empty() || unknownRule || std::from_chars(jobsText.data(), jobsText.data() + jobsText.size(), jobs).ec != std::errc{} || jobs == 0U) {
        std::println(stderr, "{}", kUsage);
        return 2;
    }
    const bool                                               list    = has("--list");
    const bool                                               summary = has("--summary");
    const bool                                               table   = has("--table");
    const std::string                                        reference{devtools::valueAfter(arguments, "--changed", "")};
    std::optional<std::map<std::string, std::set<unsigned>>> changedLines;
    if (!reference.empty()) {
        changedLines = changedLinesSince(reference, scope.roots.front());
        if (!changedLines) {
            std::println(stderr, "naming-check: git cannot tell what changed since '{}'", reference);
            return 2;
        }
    }

    const auto                                 wantedUnit    = [&](const std::string& file) { return unitPatterns.empty() ? scope.inside(file) : std::ranges::any_of(unitPatterns, [&file](const std::regex& pattern) { return std::regex_search(file, pattern); }); };
    const std::optional<std::set<std::string>> standardNames = standardNamesOf(arguments.front(), scope);
    if (!standardNames) {
        return 2;
    }

    Findings   allFindings;
    const auto unitCounts = devtools::forEachUnit(
        arguments.front(), wantedUnit, jobs, CXTranslationUnit_KeepGoing | CXTranslationUnit_DetailedPreprocessingRecord,
        [&scope, &standardNames](CXTranslationUnit unit, CXCompileCommand command) {
            Findings    findings;
            std::string firstError;
            if (hasErrors(unit, firstError)) {
                findings.unitsWithErrors.insert(std::format("{}: {}", taken(clang_CompileCommand_getFilename(command)), firstError));
                return findings; // a unit with errors has a partial syntax tree: its names are not checked
            }
            const std::map<std::string, std::set<std::string>> reflectedByRecord = reflectedMembers(unit);
            Walk                                               walk{.translationUnit = unit, .scope = &scope, .findings = &findings, .reflected = &reflectedByRecord, .standardNames = &*standardNames, .canonicalPaths = {}};
            clang_visitChildren(clang_getTranslationUnitCursor(unit), visitDeclaration, &walk);
            checkMacros(unit, scope, findings);
            return findings;
        },
        [&allFindings](Findings unitFindings) {
            allFindings.violations.merge(unitFindings.violations);
            allFindings.unitsWithErrors.merge(unitFindings.unitsWithErrors);
            for (auto& [file, names] : unitFindings.declaredIn) {
                allFindings.declaredIn[file].merge(names);
            }
            for (auto& [space, names] : unitFindings.namesInScope) {
                allFindings.namesInScope[space].merge(names);
            }
            for (auto& [enumeration, enumerators] : unitFindings.enumerators) {
                allFindings.enumerators[enumeration].merge(enumerators);
            }
            for (auto& [rule, places] : unitFindings.checked) {
                allFindings.checked[rule].merge(places);
            }
            for (auto& [ambiguity, seen] : unitFindings.observations) {
                for (auto& [what, places] : seen) {
                    allFindings.observations[ambiguity][what].merge(places);
                }
            }
        });
    if (!unitCounts) {
        std::println(stderr, "naming-check: no compile_commands.json in {}", arguments.front());
        return 2;
    }

    checkFiles(scope, allFindings);
    checkEnumeratorCases(allFindings);
    settleProposals(allFindings);
    std::vector<const Violation*> reported;
    for (const auto& [key, violation] : allFindings.violations) {
        const auto [file, line] = placeOf(violation, scope);
        const bool ruleWanted   = onlyRules.empty() || onlyRules.contains(violation.rule);
        const bool lineChanged  = !changedLines || (changedLines->contains(file) && changedLines->at(file).contains(line));
        if (ruleWanted && lineChanged) {
            reported.push_back(&violation);
        }
    }
    const auto order = [&scope](const Violation* violation) { return std::tuple{ruleNumber(violation->rule), placeOf(*violation, scope), violation->where}; };
    std::ranges::sort(reported, [&order](const Violation* left, const Violation* right) { return order(left) < order(right); });
    std::map<int, std::size_t> violationsPerRule;
    for (const Violation* violation : reported) {
        ++violationsPerRule[ruleNumber(violation->rule)];
    }
    const std::size_t failed  = unitCounts->unparsable + allFindings.unitsWithErrors.size();
    const std::string changes = reference.empty() ? std::string{} : std::format(" on lines changed since {}", reference);

    if (table) {
        if (reported.empty()) {
            std::println("No naming violations{}.", changes);
        } else {
            std::println("| where | rule | kind | name | proposed |");
            std::println("| --- | --- | --- | --- | --- |");
            for (const Violation* violation : reported) {
                std::println("| `{}` | {} | {} | `{}` | {} |", violation->where, violation->rule, violation->kind, violation->name, violation->proposal.empty() ? std::string{"—"} : std::format("`{}`", violation->proposal));
            }
        }
    } else if (summary) {
        std::println("naming: {} violations{} in {} of {} units", reported.size(), changes, unitCounts->wanted - failed, unitCounts->wanted);
        for (const auto& [number, count] : violationsPerRule) {
            const std::string id = std::format("R{}", number);
            std::println("  {}: {}", id, count);
            std::println("      {}", ruleDescription(id));
        }
    } else {
        std::println("standard-library names read: {}", standardNames->size());
        std::println("translation units checked: {} of {}", unitCounts->wanted - failed, unitCounts->wanted);
        std::println("rules (declarations checked, violations{})", changes);
        std::vector<std::pair<int, std::string>> rules;
        for (const auto& [rule, places] : allFindings.checked) {
            if (onlyRules.empty() || onlyRules.contains(rule)) {
                rules.emplace_back(ruleNumber(rule), rule);
            }
        }
        std::ranges::sort(rules);
        for (const auto& [number, rule] : rules) {
            std::println("  {}: {} checked, {} violations", rule, allFindings.checked.at(rule).size(), violationsPerRule[number]);
            std::println("      {}", ruleDescription(rule));
        }
        std::println("observations");
        for (const auto& [ambiguity, seen] : allFindings.observations) {
            for (const auto& [what, places] : seen) {
                std::println("  {} {}: {}", ambiguity, what, places.size());
                for (const std::string& place : list ? places | std::views::take(6) | std::ranges::to<std::vector>() : std::vector<std::string>{}) {
                    std::println("      {}", place);
                }
            }
        }
        std::println("std-style candidates (headers that include only the standard library and other candidates)");
        for (const std::string& candidate : stdStyleCandidates(scope)) {
            std::println("  {}", candidate);
        }
        if (list) {
            std::println("violations");
            for (const Violation* violation : reported) {
                std::println("  {} {} '{}' ({}: {}){}", violation->where, violation->kind, violation->name, violation->rule, violation->expected, violation->proposal.empty() ? "" : " -> " + violation->proposal);
            }
        }
    }
    if (failed > 0U) {
        std::println(stderr, "naming-check: {} units not checked (could not be parsed, or have errors):", failed);
        for (const std::string& unit : allFindings.unitsWithErrors) {
            std::println(stderr, "  {}", unit);
        }
        return 2;
    }
    return reported.empty() ? 0 : 1;
}
