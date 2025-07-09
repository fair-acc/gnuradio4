/// parse_registrations.cpp
/// ---------------------------------------------------------------------------
/// This parser scans a single .hpp for lines containing:
///
///   GR_REGISTER_BLOCK("OptionalQuotedName", MyTemplate, (paramPack?), [ expansions ]...)
///
/// for example:
///   GR_REGISTER_BLOCK("MyBlockName", gr::basic::Block1, ([T], [U]), [ float, double ], [int])
///   GR_REGISTER_BLOCK(gr::basic::Block0)
///   GR_REGISTER_BLOCK("blockN.hpp", gr::basic::BlockN, ([T],[U],3UZ,SomeAlgo<[T]>), [ short, int], [double])
///
/// * the GR_REGISTER_BLOCK macros must be each be on a single line (no multi-line support).
///
/// Usage:
///   parse_registrations <headerFile.hpp> <outputDir> [--split | -s]
///   e.g. parse_registrations block0.hpp build/generated
///        parse_registrations blockN.hpp build/generated --split
/// * default: each macro line => one .cpp file.
/// * --split (-s): cartesian expansion -> each block-type combination generates its own .cpp.

#include <algorithm>
#include <cstring>
#include <expected>
#include <filesystem>
#include <format>
#include <fstream>
#include <functional>
#include <iostream>
#include <ranges>
#include <string>
#include <string_view>
#include <vector>

constexpr std::string_view kRegistrationMacroName = "GR_REGISTER_BLOCK";

namespace detail {

[[nodiscard]] constexpr std::string_view trim(std::string_view sv) noexcept {
    constexpr static auto is_space = [](unsigned char c) constexpr { return c == ' ' || c == '\t' || c == '\n' || c == '\v' || c == '\f' || c == '\r'; };
    constexpr static auto notSpace = [](char c) constexpr { return !is_space(c); };

    const auto first = std::ranges::find_if(sv, notSpace);
    if (first == sv.end()) {
        return {};
    }
    const auto last = std::ranges::find_if(sv.rbegin(), sv.rend(), notSpace).base();
    return sv.substr(static_cast<std::size_t>(first - sv.begin()), static_cast<std::size_t>(last - first));
}

constexpr std::expected<std::vector<std::string_view>, std::string> splitTopLevelCommaSeparatedValues(std::string_view input) {
    std::vector<std::string_view> tokens;
    std::vector<char>             bracketStack;
    std::size_t                   start = 0;

    input = trim(input);
    for (std::size_t i = 0; i < input.size(); ++i) {
        char c = input[i];
        if (c == '(' || c == '[') {
            bracketStack.push_back(c == '(' ? ')' : ']');
        } else if (c == ')' || c == ']') {
            if (bracketStack.empty() || bracketStack.back() != c) {
                return std::unexpected("mismatched bracket");
            }
            bracketStack.pop_back();
        } else if (c == ',' && bracketStack.empty()) {
            tokens.push_back(trim(input.substr(start, i - start)));
            start = i + 1;
        }
    }

    if (!bracketStack.empty()) {
        return std::unexpected("mismatched bracket");
    }

    tokens.push_back(trim(input.substr(start)));
    return tokens;
}

} // namespace detail

struct Options {
    std::filesystem::path headerPath;
    std::filesystem::path outDir;
    std::string           registryHeader   = "gnuradio-4.0/BlockRegistry.hpp";
    std::string           registryInstance = "gr::globalBlockRegistry";
    bool                  expansionsSplit  = false; // true: each block instantiation creates its own file

    Options(int argc, char** argv) {
        headerPath = argv[1];
        outDir     = argv[2];
        for (int index = 3; index < argc; index++) {
            if ((std::strcmp(argv[index], "--split") == 0) || (std::strcmp(argv[index], "-s") == 0)) {
                expansionsSplit = true;
            } else if (argc > index + 1 && (std::strcmp(argv[index], "--registry-header") == 0)) {
                registryHeader = argv[index + 1];
            } else if (argc > index + 1 && (std::strcmp(argv[index], "--registry-instance") == 0)) {
                registryInstance = argv[index + 1];
            }
        }
    }
};

struct RegisterBlock {
    std::string_view                           baseName;     // e.g. "MyBlock" if quoted
    std::string_view                           templateName; // e.g. "gr::basic::Block1"
    std::string_view                           paramPack;    // e.g. "([T], [U])"
    std::vector<std::vector<std::string_view>> expansions;   // e.g. [float, double], [int] => expansions = { {"float","double"}, {"int"} }
};

std::ofstream openFile(std::string fileName) {
    std::cout << std::format("\t=> Generating file: '{}'\n", fileName);
    std::ofstream result(fileName);
    if (!result.is_open()) {
        throw std::format("Failed to open '{}' for writing", fileName);
    }
    return result;
}

struct GeneratedFiles {
    std::ofstream registrations;
    std::ofstream declarations;
    std::ofstream rawCalls;

    GeneratedFiles(std::string outFileBase)
        : registrations(openFile(outFileBase + ".cpp")),      //
          declarations(outFileBase + "_declarations.hpp.in"), //
          rawCalls(outFileBase + "_raw_calls.hpp.in") {}
};

// Returns the name of the generated function
[[nodiscard]] std::string emitRegistrationCode(auto& registrationsOutput, std::string_view namePrefix, std::size_t hashValue, std::string_view templateName, std::string_view finalName, std::size_t lineNum, const Options& options) {
    registrationsOutput << std::format(R"cppcode(
    namespace gr {{ class BlockRegistry; }}
    namespace {{
        bool reg_{}_{}(gr::BlockRegistry& registry) {{
            return gr::registerBlock<{}, "{}">(registry); // for details: {}:{}\n"
        }}
    }}
)cppcode", //
        namePrefix, hashValue, templateName, finalName.empty() ? "" : finalName, options.registryInstance, options.headerPath.string(), lineNum);
    return std::format("reg_{}_{}", namePrefix, hashValue);
}

[[nodiscard]] std::string emitInitAllCode(auto& fout, std::string_view namePrefix, const std::vector<std::string>& generatedRegistrationFunctions, const Options& options) {
    fout << std::format("\nextern \"C\" {{\nGNURADIO_EXPORT std::size_t gr_blocklib_init_unit_{}(gr::BlockRegistry& registry) {{\n    std::size_t result = 0UZ; \n", namePrefix);
    for (const auto& generatedRegistrationFunction : generatedRegistrationFunctions) {
        fout << "    result += ( !" << generatedRegistrationFunction << "(registry) ? 1UZ : 0UZ );\n";
    }
    fout << "    return result;\n}\n}\n\n";

    fout << std::format("auto gr_blocklib_init_unit_{0}_invoked = gr_blocklib_init_unit_{0}({1}());\n", namePrefix, options.registryInstance);
    return std::format("gr_blocklib_init_unit_{}", namePrefix);
}

static std::expected<RegisterBlock, std::string> parseRegisterBlockMacro(std::string_view line) {
    auto macroPos = line.find(kRegistrationMacroName);
    if (macroPos == std::string_view::npos) {
        return std::unexpected("no macro found on this line.");
    }

    std::size_t open = line.find('(', macroPos + kRegistrationMacroName.size());
    if (open == std::string::npos) {
        return std::unexpected("missing '(' after macro");
    }
    std::size_t close = line.rfind(')');
    if ((close == std::string::npos) || (close <= open)) {
        return std::unexpected("missing ')' after macro");
    }

    // content of GR_REGISTER_BLOCK(...)
    std::string_view macroContent = detail::trim(line.substr(open + 1, close - open - 1));
    auto             exParts      = detail::splitTopLevelCommaSeparatedValues(macroContent);
    if (!exParts.has_value()) {
        return std::unexpected("erroneous or empty macro body?");
    }
    auto parts = exParts.value();

    RegisterBlock rb;
    std::size_t   idx = 0;

    // if first part is quoted => user-defined baseName
    if ((parts[idx].size() >= 2) && (parts[idx].front() == '"') && (parts[idx].back() == '"')) { // remove outer quotes
        rb.baseName = {parts[idx].data() + 1, parts[idx].size() - 2};
        idx++;
    }

    // parse template base name
    if (idx >= parts.size()) {
        return std::unexpected("missing templateName argument");
    }
    rb.templateName = detail::trim(parts[idx]);
    idx++;

    // parse template type/NTTP parameter list
    if (idx < parts.size()) {
        rb.paramPack = detail::trim(parts[idx]);
        idx++;
    }

    // remainder: parse specific template types [T] that need to be instantiated, e.g. "[ int, float, double, ], [ int, short], ..."
    for (; idx < parts.size(); idx++) {
        auto chunk = detail::trim(parts[idx]);
        if ((chunk.size() >= 2) && (chunk.front() == '[') && (chunk.back() == ']')) {
            chunk = detail::trim(chunk.substr(1, chunk.size() - 2));
        }
        auto exSub = detail::splitTopLevelCommaSeparatedValues(chunk);
        if (!exSub.has_value()) {
            return std::unexpected(std::format("couldn't parse '{}' error: {}", chunk, exSub.error()));
        }
        std::vector<std::string_view> sub = exSub.value();
        std::vector<std::string_view> group;
        group.reserve(sub.size());
        for (const auto& sv : sub) {
            auto val = detail::trim(sv);
            if (!val.empty()) {
                group.emplace_back(val);
            }
        }
        if (!group.empty()) {
            rb.expansions.push_back(std::move(group));
        }
    }

    return rb;
}

// cartesianProduct => expansions => combos
static std::vector<std::vector<std::string_view>> cartesianProduct(const std::vector<std::vector<std::string_view>>& groups) {
    if (groups.empty()) {
        return {{}};
    }
    std::vector<std::vector<std::string_view>> result{{}};
    for (auto& g : groups) {
        std::vector<std::vector<std::string_view>> temp;
        for (const auto& prefix : result) {
            for (auto& val : g) {
                auto row = prefix;
                row.push_back(val);
                temp.push_back(std::move(row));
            }
        }
        result = std::move(temp);
    }
    return result;
}

// replacePlaceholders => paramPack="([T],[U])" => expansions => "float,int"
static std::string replacePlaceholders(std::string param, const std::vector<std::string_view>& vars) {
    // remove leading '(' or ' ' and trailing ')' or ' '
    while (!param.empty() && ((param.front() == '(') || (param.front() == ' '))) {
        param.erase(param.begin());
    }
    while (!param.empty() && ((param.back() == ')') || (param.back() == ' '))) {
        param.pop_back();
    }

    constexpr std::array placeholders = {"[T]", "[U]", "[A]", "[B]", "[X]", "[Y]", "[Z]", "[S]"};
    for (int i = 0; (i < placeholders.size()) && (i < static_cast<int>(vars.size())); i++) {
        auto&       val = vars[i];
        auto        ph  = placeholders[i];
        std::size_t pos = 0;
        while ((pos = param.find(ph, pos)) != std::string::npos) {
            param.replace(pos, std::strlen(ph), val);
            pos += val.size();
        }
    }
    return param;
}

int main(int argc, char** argv) try {
    std::filesystem::path commandPath = argv[0];
    if (argc < 3) {
        std::cerr << std::format("Usage: {} <header.hpp> <outputDir> [--split | -s] [--registry-header include_file.hpp]\n", commandPath.string());
        return 1;
    }

    Options options(argc, argv);

    std::cout << std::format("parsing header: '{}' -> '{}'  split: {} \n", options.headerPath.string(), options.outDir.string(), options.expansionsSplit ? "Yes" : "No");

    if (!std::filesystem::exists(options.headerPath)) {
        std::cerr << std::format("error: file '{}' not found.\n", options.headerPath.string());
        return 1;
    }

    std::ifstream fin(options.headerPath);
    if (!fin.is_open()) {
        throw std::format("cannot open '{}' for writing", options.headerPath.string());
    }

    std::filesystem::create_directories(options.outDir);

    auto stem       = options.headerPath.stem().string();
    int  macroCount = 0UZ;
    int  fileCount  = 0UZ;

    const auto moduleName = options.outDir.filename().string();

    const auto integratorSourceFile = (options.outDir / "integrator.cpp");
    if (!std::filesystem::exists(integratorSourceFile)) {
        std::ofstream integrator = openFile(integratorSourceFile.string());
        integrator << std::format(R"cppcode(
            #include <gnuradio-4.0/BlockRegistry.hpp>

            #include "declarations.hpp"

            extern "C" {{
                GNURADIO_EXPORT
                std::size_t gr_blocklib_init_module_{}(gr::BlockRegistry& registry) {{
                    std::size_t result = 0UZ;
                    #include "raw_calls.hpp"
                    return result;
                }}
            }}
        )cppcode",
            moduleName);
    }

    const auto integratorHeaderFile = (options.outDir / (moduleName + ".hpp"));
    if (!std::filesystem::exists(integratorHeaderFile)) {
        std::ofstream integrator = openFile(integratorHeaderFile.string());
        integrator << std::format(R"cppcode(
            #ifndef GR_BLOCKLIB_INIT_MODULE_{0}
            #define GR_BLOCKLIB_INIT_MODULE_{0}
            namespace gr {{ class BlockRegistry; }}

            extern "C" {{
                GNURADIO_EXPORT
                std::size_t gr_blocklib_init_module_{0}(gr::BlockRegistry& registry);
            }}

            namespace gr::blocklib {{
                inline
                std::size_t init{0}(gr::BlockRegistry& registry) {{
                    return gr_blocklib_init_module_{0}(registry);
                }}
            }}
            #endif
        )cppcode",
            moduleName);
    }

    std::string line;
    std::size_t lineNum = 0UZ;
    while (std::getline(fin, line)) {
        lineNum++;
        auto trimmed = detail::trim(line);
        if (trimmed.empty() || trimmed.starts_with("//") || !trimmed.contains(kRegistrationMacroName)) {
            continue;
        }

        std::cout << std::format("\tfound macro on line {}: '{}'\n", lineNum, trimmed);

        auto maybe = parseRegisterBlockMacro(trimmed);
        if (!maybe) {
            std::cerr << std::format("\terror: parse failure in {}:{} => {}\n", options.headerPath.filename().string(), lineNum, maybe.error());
            continue;
        }
        auto& info = *maybe;

        // we do 1 .cpp per macro, or multiple if expansionsSplit => each expansion => a new .cpp
        if (auto combos = cartesianProduct(info.expansions); !options.expansionsSplit || combos.empty()) {
            std::vector<std::string> generatedRegistrationFunctions;

            // Single .cpp for all expansions of this macro
            const auto namePrefix  = std::format("{}_{}", stem, macroCount);
            const auto outFileBase = (options.outDir / namePrefix).string();

            GeneratedFiles output(outFileBase);

            output.registrations << std::format("// auto-generated by {}, do not edit.\n", commandPath.string());
            output.registrations << std::format("#include <{1}>\n#include \"{0}\" // for details: {0}:1\n\n", options.headerPath.string(), options.registryHeader);

            if (combos.empty()) {
                // No expansions => single call
                auto        replaced = replacePlaceholders(std::string(info.paramPack), {});
                std::string finalName;
                if (!info.baseName.empty() && !replaced.empty()) {
                    finalName = std::format("{}<{}>", info.baseName, replaced);
                } else {
                    finalName = std::string(info.baseName);
                }

                const std::string templateName = std::format("{}{}", info.templateName, replaced.empty() ? "" : std::format("<{}>", replaced));
                const std::size_t hashValue    = std::hash<std::string>{}(templateName);
                generatedRegistrationFunctions.push_back(emitRegistrationCode(output.registrations, namePrefix, hashValue, templateName, finalName, lineNum, options));
            } else {
                // multiple expansions => all in one file
                int localIdx = 0;
                for (const auto& vars : combos) {
                    auto        replaced = replacePlaceholders(std::string(info.paramPack), vars);
                    std::string finalName;
                    if (!info.baseName.empty() && !replaced.empty()) {
                        finalName = std::format("{}<{}>", info.baseName, replaced);
                    } else {
                        finalName = std::string(info.baseName);
                    }

                    const std::string templateName = std::format("{}{}", info.templateName, replaced.empty() ? "" : std::format("<{}>", replaced));
                    const std::size_t hashValue    = std::hash<std::string>{}(templateName);
                    generatedRegistrationFunctions.push_back(emitRegistrationCode(output.registrations, namePrefix, hashValue, templateName, finalName, lineNum, options));
                    localIdx++;
                }
            }

            auto generatedInitFunction = emitInitAllCode(output.registrations, namePrefix, generatedRegistrationFunctions, options);
            output.registrations << "// To initialize, call " << generatedInitFunction << "\n";

            const std::string declarationsGuard = std::format("HEADER_GUARD_{}_HPP", generatedInitFunction);
            output.declarations << std::format("#ifndef {}\n#define {}\n", declarationsGuard, declarationsGuard);
            output.declarations << "extern \"C\" { std::size_t " << generatedInitFunction << "(gr::BlockRegistry&); }\n";
            output.declarations << std::format("#endif // {}\n", declarationsGuard);
            output.rawCalls << "result += !" << generatedInitFunction << "(registry);\n";

            output.registrations << "// end of auto-generated code\n";
            fileCount++;
        } else {
            // expansionsSplit => each expansion => its own file
            int localIdx = 0;
            for (const auto& vars : combos) {
                std::vector<std::string> generatedRegistrationFunctions;

                const auto replaced    = replacePlaceholders(std::string(info.paramPack), vars);
                const auto namePrefix  = std::format("{}_{}_{}", stem, macroCount, localIdx);
                const auto outFileBase = (options.outDir / namePrefix).string();

                GeneratedFiles output(outFileBase);

                output.registrations << std::format("// auto-generated by {}, do not edit.\n", commandPath.string());
                output.registrations << std::format("#include <{1}>\n#include \"{0}\" // for details: {0}:1\n\n", options.headerPath.string(), options.registryHeader);

                std::string finalName;
                if (!info.baseName.empty() && !replaced.empty()) {
                    finalName = std::format("{}<{}>", info.baseName, replaced);
                } else {
                    finalName = std::string(info.baseName);
                }

                const std::string templateName = std::format("{}{}", info.templateName, replaced.empty() ? "" : std::format("<{}>", replaced));
                const std::size_t hashValue    = std::hash<std::string>{}(templateName);
                generatedRegistrationFunctions.push_back(emitRegistrationCode(output.registrations, namePrefix, hashValue, templateName, finalName, lineNum, options));
                auto generatedInitFunction = emitInitAllCode(output.registrations, namePrefix, generatedRegistrationFunctions, options);
                output.registrations << "// To initialize, call " << generatedInitFunction << "\n";

                const std::string declarationsGuard = std::format("HEADER_GUARD_{}_HPP", generatedInitFunction);
                output.declarations << std::format("#ifndef {}\n#define {}\n", declarationsGuard, declarationsGuard);
                output.declarations << "extern \"C\" { bool " << generatedInitFunction << "(gr::BlockRegistry&); }\n";
                output.declarations << std::format("#endif // {}\n", declarationsGuard);
                output.rawCalls << "result += !" << generatedInitFunction << "(registry);\n";

                output.registrations << "// end of auto-generated code\n";

                fileCount++;
                localIdx++;
            }
        }

        macroCount++;
        std::cout << std::endl;
    }

    std::cout << std::format("parse_registrations: Wrote {} file(s) for {} macro definition(s).\n", fileCount, macroCount);
    return 0;
} catch (const std::string& error) {
    std::cerr << "ERROR: " << error << '\n';
    return EXIT_FAILURE;
}
