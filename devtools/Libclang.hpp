#ifndef GNURADIO_DEVTOOLS_LIBCLANG_HPP
#define GNURADIO_DEVTOOLS_LIBCLANG_HPP

// the libclang plumbing devtools/code-metrics and devtools/naming-check share: a build's compile commands, its
// translation units parsed on several threads, and the conversions every cursor walk needs

#include <clang-c/CXCompilationDatabase.h>
#include <clang-c/Index.h>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <filesystem>
#include <functional>
#include <iterator>
#include <mutex>
#include <optional>
#include <print>
#include <set>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace devtools {

inline std::string taken(CXString libclangText) {
    std::string ownedText = clang_getCString(libclangText) == nullptr ? std::string{} : clang_getCString(libclangText);
    clang_disposeString(libclangText);
    return ownedText;
}

inline std::string fileOf(CXSourceLocation location, unsigned* line = nullptr) {
    CXFile file = nullptr;
    clang_getExpansionLocation(location, &file, line, nullptr, nullptr);
    return file == nullptr ? std::string{} : std::filesystem::weakly_canonical(taken(clang_getFileName(file))).string();
}

inline CXCursor firstChildOf(CXCursor cursor) {
    CXCursor firstChild = clang_getNullCursor();
    clang_visitChildren(
        cursor,
        [](CXCursor child, CXCursor, CXClientData firstChildSlot) {
            *static_cast<CXCursor*>(firstChildSlot) = child;
            return CXChildVisit_Break;
        },
        &firstChild);
    return firstChild;
}

inline std::string_view valueAfter(const std::vector<std::string_view>& arguments, std::string_view flag, std::string_view fallback) {
    const auto flagPosition = std::ranges::find(arguments, flag);
    return flagPosition != arguments.end() && flagPosition + 1 != arguments.end() ? *(flagPosition + 1) : fallback;
}

/// the compiler arguments of the first unit `wanted` accepts in `buildDirectory`'s compile_commands.json, with its own
/// source file named `source` instead, so a scratch file is parsed exactly as the build parses its code
inline std::vector<std::string> argumentsFor(std::string_view buildDirectory, const std::function<bool(const std::string&)>& wanted, const std::string& source) {
    CXCompilationDatabase_Error loadError = CXCompilationDatabase_NoError;
    CXCompilationDatabase       database  = clang_CompilationDatabase_fromDirectory(std::string{buildDirectory}.c_str(), &loadError);
    std::vector<std::string>    arguments;
    if (loadError != CXCompilationDatabase_NoError) {
        return arguments;
    }
    CXCompileCommands commands = clang_CompilationDatabase_getAllCompileCommands(database);
    for (unsigned commandIndex = 0; commandIndex < clang_CompileCommands_getSize(commands) && arguments.empty(); ++commandIndex) {
        CXCompileCommand  command    = clang_CompileCommands_getCommand(commands, commandIndex);
        const std::string sourceFile = taken(clang_CompileCommand_getFilename(command));
        if (!wanted(std::filesystem::weakly_canonical(sourceFile).string())) {
            continue;
        }
        for (unsigned argumentIndex = 0; argumentIndex < clang_CompileCommand_getNumArgs(command); ++argumentIndex) {
            std::string argument = taken(clang_CompileCommand_getArg(command, argumentIndex));
            arguments.push_back(argument == sourceFile ? source : argument);
        }
    }
    clang_CompileCommands_dispose(commands);
    clang_CompilationDatabase_dispose(database);
    return arguments;
}

struct UnitCounts {
    std::size_t wanted     = 0; // units `wanted` accepted
    std::size_t unparsable = 0; // of these, units libclang could not parse at all: not analysed
};

/// parses every unit of `buildDirectory`'s compile_commands.json whose file `wanted` accepts, each file once, on `jobs`
/// threads with libclang `options`; `analyse(unit, command)` runs on a worker, `merge` folds its result in under a
/// lock; nothing when the build has no compile_commands.json
template<typename TAnalyse, typename TMerge>
std::optional<UnitCounts> forEachUnit(std::string_view buildDirectory, const std::function<bool(const std::string&)>& wanted, std::size_t jobs, unsigned options, TAnalyse analyse, TMerge merge) {
    CXCompilationDatabase_Error loadError = CXCompilationDatabase_NoError;
    CXCompilationDatabase       database  = clang_CompilationDatabase_fromDirectory(std::string{buildDirectory}.c_str(), &loadError);
    if (loadError != CXCompilationDatabase_NoError) {
        return std::nullopt;
    }
    CXCompileCommands             commands = clang_CompilationDatabase_getAllCompileCommands(database);
    std::vector<CXCompileCommand> wantedCommands;
    std::set<std::string>         wantedSources;
    for (unsigned commandIndex = 0; commandIndex < clang_CompileCommands_getSize(commands); ++commandIndex) {
        CXCompileCommand  command    = clang_CompileCommands_getCommand(commands, commandIndex);
        const std::string sourceFile = std::filesystem::weakly_canonical(taken(clang_CompileCommand_getFilename(command))).string();
        if (wanted(sourceFile) && wantedSources.insert(sourceFile).second) {
            wantedCommands.push_back(command);
        }
    }

    std::mutex               mergeLock;
    std::atomic<std::size_t> nextUnit{0};
    std::atomic<std::size_t> unparsable{0};
    {
        std::vector<std::jthread> workers;
        for (std::size_t worker = 0; worker < std::min(jobs, wantedCommands.size()); ++worker) {
            workers.emplace_back([&] {
                CXIndex clangIndex = clang_createIndex(0, 0);
                for (std::size_t commandIndex = nextUnit++; commandIndex < wantedCommands.size(); commandIndex = nextUnit++) {
                    std::vector<std::string> arguments;
                    for (unsigned argumentIndex = 0; argumentIndex < clang_CompileCommand_getNumArgs(wantedCommands[commandIndex]); ++argumentIndex) {
                        arguments.push_back(taken(clang_CompileCommand_getArg(wantedCommands[commandIndex], argumentIndex)));
                    }
                    std::vector<const char*> argumentPointers;
                    std::ranges::transform(arguments, std::back_inserter(argumentPointers), &std::string::c_str);
                    CXTranslationUnit translationUnit = nullptr;
                    if (clang_parseTranslationUnit2FullArgv(clangIndex, nullptr, argumentPointers.data(), static_cast<int>(argumentPointers.size()), nullptr, 0, options, &translationUnit) != CXError_Success) {
                        std::println(stderr, "cannot parse {}", taken(clang_CompileCommand_getFilename(wantedCommands[commandIndex])));
                        ++unparsable;
                        continue;
                    }
                    auto unitFindings = analyse(translationUnit, wantedCommands[commandIndex]);
                    clang_disposeTranslationUnit(translationUnit);
                    const std::scoped_lock lock{mergeLock};
                    merge(std::move(unitFindings));
                }
                clang_disposeIndex(clangIndex);
            });
        }
    }
    const UnitCounts counts{.wanted = wantedCommands.size(), .unparsable = unparsable.load()};
    clang_CompileCommands_dispose(commands);
    clang_CompilationDatabase_dispose(database);
    return counts;
}

} // namespace devtools

#endif // GNURADIO_DEVTOOLS_LIBCLANG_HPP
