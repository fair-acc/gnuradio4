// compile with: clang++-18 check_naming_convention.cpp -o check_naming_convention `llvm-config --cxxflags --ldflags --libs --system-libs` -fexceptions -lclang-cpp
#include <clang/AST/ASTConsumer.h>
#include <clang/AST/RecursiveASTVisitor.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Frontend/FrontendActions.h>
#include <clang/Tooling/CommonOptionsParser.h>
#include <clang/Tooling/Tooling.h>
#include <llvm/Support/CommandLine.h>

#include <fstream>
#include <iostream>
#include <map>
#include <regex>
#include <sstream>
#include <string>
#include <string_view>

constexpr std::string_view kRed    = "\033[31m";
constexpr std::string_view kGreen  = "\033[32m";
constexpr std::string_view kOrange = "\033[38;5;208m";
constexpr std::string_view kReset  = "\033[0m";

bool
consoleSupportsColor() {
    const static char *kTerminal      = getenv("TERM");
    const static bool  kSupportsColor = kTerminal && std::string(kTerminal) != "dumb";
    return kSupportsColor;
}

std::string
colorWrap(const std::string &text, const std::string_view &color) {
    if (consoleSupportsColor()) {
        return std::string(color) + text + std::string(kReset);
    } else {
        return text;
    }
}

std::string
colorSubstring(const std::string &haystack, const std::string &needle, const std::string_view &color, bool underline = true) {
    std::string result;
    size_t      pos = 0;
    size_t      foundPos;
    while ((foundPos = haystack.find(needle, pos)) != std::string::npos) {
        // Append the part of haystack before the needle, plus the colored needle
        result += haystack.substr(pos, foundPos - pos) + colorWrap(needle, color);
        pos = foundPos + needle.length();
    }
    // Append the remaining part of the haystack
    result += haystack.substr(pos);
    return result;
}

using namespace clang;
using namespace clang::tooling;
using namespace llvm;

void
validateAstDeclaration(const std::string &name, const std::string &type, const std::string &location, const std::string &accessSpecifier, bool isConst, const std::string &contextLines);

class FindNamedDeclVisitor : public RecursiveASTVisitor<FindNamedDeclVisitor> {
    ASTContext *_context;

public:
    explicit FindNamedDeclVisitor(ASTContext *context) : _context(context) {}

    bool
    VisitNamedDecl(NamedDecl *astDeclaration) { // NOSONAR public clang method interface
        if (!astDeclaration) return true;

        SourceManager &sourceManager = _context->getSourceManager();
        if (sourceManager.isInSystemHeader(astDeclaration->getLocation())) return true;

        auto location = sourceManager.getPresumedLoc(astDeclaration->getLocation());

        if (!location.isValid()) return true; // skip invalid locations

        if (!sourceManager.isInMainFile(astDeclaration->getLocation())) return true; // skip astDeclarations not in the main file

        const auto removeDeclSuffix = [](const std::string &declType) { return declType.substr(0, declType.size() - 4 * (declType.size() >= 4 && declType.rfind("Decl") == declType.size() - 4)); };

        std::string declName     = astDeclaration->getNameAsString();
        std::string declType     = removeDeclSuffix(astDeclaration->getDeclKindName());
        std::string declLocation = std::string(location.getFilename()) + ":" + std::to_string(location.getLine());

        // Access specifier
        std::string accessSpecifier;
        if (const auto *recordDecl = dyn_cast<CXXRecordDecl>(astDeclaration->getDeclContext())) {
            AccessSpecifier as = astDeclaration->getAccess();
            switch (as) {
            case AS_private: accessSpecifier = "private"; break;
            case AS_protected: accessSpecifier = "protected"; break;
            case AS_public: accessSpecifier = "public"; break;
            default: accessSpecifier = ""; // not applicable
            }
        }

        bool isConst = false;
        if (const VarDecl *variableDeclaration = dyn_cast<VarDecl>(astDeclaration)) {
            if (variableDeclaration->getType().isConstQualified()) {
                isConst = true;
            }
        }

        unsigned  declLine    = sourceManager.getExpansionLineNumber(astDeclaration->getLocation());
        StringRef fileContent = sourceManager.getBufferData(sourceManager.getFileID(astDeclaration->getLocation()));

        auto containsDisableCheck = [](std::string_view str, std::string_view pattern) {
            auto it = std::search(str.begin(), str.end(), pattern.begin(), pattern.end(), [](char ch1, char ch2) { return std::tolower(ch1) == std::tolower(ch2); });
            return it != str.end();
        };

        std::istringstream stream(fileContent.str());
        std::string        line;
        std::string        contextLines;
        unsigned           currentLine = 0;
        bool               skipLine    = false;
        while (std::getline(stream, line)) {
            ++currentLine;
            if (currentLine >= declLine - 1 && currentLine <= declLine + 1) {
                if (currentLine == declLine) {
                    if (containsDisableCheck(line, "nosonar")) { // disable naming check for this line
                        skipLine = true;
                    }
                    if ((declType == "VarTemplate" || declType == "Var")) {
                        // special case to interpret the 'concept' keyword and rule since the clang AST
                        // does not seem to distinguish it from a template or variable.
                        auto startsWithConcept = [](const std::string &str) {
                            auto pos = str.find_first_not_of(" \t\n\r\f\v");
                            return pos != std::string::npos && str.substr(pos, 7) == "concept";
                        };
                        if (startsWithConcept(line)) { // detected concept
                            declType = "Concept";
                        }
                    }
                }
                contextLines += std::to_string(currentLine) + ": " + line + "\n";
            }
        }

        if (!skipLine) {
            validateAstDeclaration(declName, declType, declLocation, accessSpecifier, isConst, contextLines);
        }

        return true;
    }
};

class FindNamedDeclConsumer : public ASTConsumer {
    FindNamedDeclVisitor _visitor;

public:
    explicit FindNamedDeclConsumer(ASTContext *context) : _visitor(context) {}

    void
    HandleTranslationUnit(ASTContext &context) override { // NOSONAR naming rule does not apply for clang public API
        _visitor.TraverseDecl(context.getTranslationUnitDecl());
    }
};

struct FindNamedDeclAction : public ASTFrontendAction {
    std::unique_ptr<ASTConsumer>
    CreateASTConsumer(CompilerInstance &ci, StringRef file) override { // NOSONAR naming rule does not apply for clang public API
        return std::make_unique<FindNamedDeclConsumer>(&ci.getASTContext());
    }
};

struct NamingConvention {
    std::vector<std::pair<std::string, std::regex>> regexes; // Pair of pattern string and compiled regex
    std::string                                     accessModifier;
    bool                                            isConst = false;
    std::string                                     description;
};

std::multimap<std::string, NamingConvention> namingConventions;
int                                          foundViolations = 0;

[[nodiscard]] std::string
trim(std::string s) {
    constexpr auto predicate = [](unsigned char ch) { return !std::isspace(ch); };
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), predicate));
    s.erase(std::find_if(s.rbegin(), s.rend(), predicate).base(), s.end());
    return s;
}

void
loadNamingConventions(const std::string &filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "No naming convention file found. Using default conventions.\n";
        return;
    }

    std::string line;
    while (std::getline(file, line)) {
        auto isCommented = [](const std::string &str) {
            auto pos = str.find_first_not_of(" \t\n\r\f\v");
            return pos != std::string::npos && str[pos] == '#';
        };
        if (isCommented(line)) {
            continue;
        }
        std::istringstream       iss(line);
        std::string              token;
        std::vector<std::string> tokens;

        while (std::getline(iss, token, ':')) {
            tokens.push_back(trim(token));
        }

        if (tokens.size() < 2) continue; // Not enough tokens

        std::string      type = tokens[0];
        NamingConvention convention;
        for (size_t i = 1; i < tokens.size(); i++) {
            if (tokens[i] == "public" || tokens[i] == "private" || tokens[i] == "protected") {
                convention.accessModifier = tokens[i];
            } else if (tokens[i] == "const") {
                convention.isConst = true;
            } else if (i == tokens.size() - 1 && tokens[i].find_first_of("^[") == std::string::npos) {
                // Assume last token might be a description if it doesn't look like a regex
                convention.description = tokens[i];
            } else {
                try {
                    convention.regexes.emplace_back(tokens[i], std::regex(tokens[i], std::regex::ECMAScript));
                } catch (const std::regex_error &e) {
                    llvm::errs() << "Regex error in pattern: " << tokens[i] << " - " << e.what() << '\n';
                }
            }
        }
        namingConventions.emplace(type, convention);
    }
}

void
validateAstDeclaration(const std::string &name, const std::string &type, const std::string &location, const std::string &accessSpecifier, bool isConst, const std::string &contextLines) {
    auto range = namingConventions.equal_range(type);

    if (range.first == range.second) {
        std::cout << "found undefined type: " << type << " for entity:" << name << "\n";
        return;
    }

    struct MatchScore {
        int                     regexScore  = 0;
        int                     accessScore = 0;
        int                     constScore  = 0;
        const NamingConvention *convention  = nullptr;
    };

    MatchScore bestMatch;
    for (auto it = range.first; it != range.second; ++it) {
        const NamingConvention &convention = it->second;
        MatchScore              currentScore;
        currentScore.convention = &convention;

        // check against all regex patterns in the naming convention
        for (const auto &regex : convention.regexes) {
            if (std::regex_match(name, regex.second)) {
                currentScore.regexScore = 1;
                break;
            }
        }

        currentScore.accessScore = (convention.accessModifier.empty() || convention.accessModifier == accessSpecifier) ? 1 : 0;
        currentScore.constScore  = (convention.isConst == isConst) ? 1 : 0;

        int totalScore     = currentScore.regexScore + currentScore.accessScore + currentScore.constScore;
        int bestTotalScore = bestMatch.regexScore + bestMatch.accessScore + bestMatch.constScore;

        if (totalScore >= 3) {
            return; // perfect match is found
        }

        if (totalScore > bestTotalScore) {
            bestMatch = currentScore;
        }
    }

    if (bestMatch.convention) {
        std::cout << "non-conformity: " << location << " for type: " << colorWrap(type, kOrange) << ", name: " << colorWrap(name, kRed) << " - closest matching rule: ";
        for (const auto &regex : bestMatch.convention->regexes) {
            std::cout << regex.first << " "; // Output the pattern string
        }
        if (!bestMatch.convention->description.empty()) {
            std::cout << "rule: " << colorWrap(bestMatch.convention->description, kGreen);
        }
        std::cout << "\n" << colorSubstring(contextLines, name, kRed, true) << "\n\n";
        foundViolations++;
    } else {
        std::cout << "No valid naming conventions found for " << type << " that match all criteria.\n";
    }
}

int
main(int argc, const char **argv) {
    loadNamingConventions(".namingConvention");

    auto expectedParser = CommonOptionsParser::create(argc, argv, llvm::cl::getGeneralCategory());
    if (!expectedParser) {
        llvm::errs() << expectedParser.takeError();
        return 1;
    }
    CommonOptionsParser &optionsParser = expectedParser.get();

    ClangTool tool(optionsParser.getCompilations(), optionsParser.getSourcePathList());

    tool.run(newFrontendActionFactory<FindNamedDeclAction>().get());

    if (foundViolations == 0) {
        std::cout << colorWrap("no naming nonconformities found - very good!\n", kGreen);
    } else {
        std::cerr << colorWrap("found ", kRed) << colorWrap(std::to_string(foundViolations), kRed) << colorWrap(" naming nonconformities\n", kRed);
    }

    return foundViolations;
}
