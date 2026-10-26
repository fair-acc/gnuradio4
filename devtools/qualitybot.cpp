// Per bot one sticky PR comment, one status and one history row on the `ci-metrics` branch, from the tools' own output.
// usage: qualitybot <bot> --out <dir>: comment.md, status.json and figures.csv, which .github/actions/report posts

#include "QualityBot.hpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <print>
#include <random>
#include <regex>
#include <sstream>
#include <thread>

#include <sys/wait.h>
#include <unistd.h>

namespace fs = std::filesystem;
using namespace devtools::bot;

namespace {

struct Settings {
    std::string                                                   historyRef      = "origin/ci-metrics";
    std::string                                                   marker          = "gr4";
    std::string                                                   author          = "gnuradio4 metrics";
    std::size_t                                                   chartPoints     = 100;
    long long                                                     chartMaxAgeDays = 365;
    std::size_t                                                   tableRows       = 30;
    std::size_t                                                   diffLines       = 300;
    std::size_t                                                   keptReports     = 30;
    std::vector<std::pair<std::string, std::vector<std::string>>> formatters      = {{"clang-format", {".cpp", ".hpp", ".h", ".cc", ".cxx"}}, {"cmake-format", {"CMakeLists.txt", ".cmake"}}, {"prettier", {".md", ".yml", ".yaml"}}, {"black", {".py"}}, {"shellcheck", {".sh"}}};
    std::vector<std::string>                                      namingArguments = {"--root", "core", "--root", "meta", "--root", "algorithm", "--root", "blocks", "--root", "devtools", "--exclude", "blocks/testing", "--exclude", "devtools/test/naming-check", "--exclude", "devtools/test/code-metrics", "--std-style", "gr::meta", "--std-style", "gr::thread_pool", "--std-style", "gr::math", "--std-style", "gr::allocator", "--std-style", "gr::pmr"};
    // libclang parses a heavy GR4 unit in about 3.3 GB (qa_Scheduler, 2026-10-15): three fit a 16 GB runner
    std::size_t parseJobs = 3;
    // the sums CLAUDE.md (measured quality) names; ratios move with any deletion
    std::vector<std::string>                         codeStyleCounted = {"cognitive complexity", "split-invariant complexity", "tokens + comment words", "repeated expressions", "nested ternaries", "write-only fields", "unused functions", "flag arguments", "restating comments", "lines over 160 characters"};
    std::string                                      codeStyleChart   = "cognitive complexity";
    std::vector<std::pair<std::string, std::string>> coverageAreas    = {{"core", "core/"}, {"blocks", "blocks/"}, {"algorithm", "algorithm/"}, {"meta", "meta/"}};
    std::string                                      coverageExcluded = R"((/_deps/|/test/|/usr/|/build/|/third_party/|/blocks/testing/|/blocks/[^/]+/src/|/algorithm/src/))";
    // bumped by hand when the instrumentation flags in checks.yml change: they are not in a file of the tool list
    std::string              coverageMethod  = "acpp-omp/llvm-cov-21/2";
    double                   patchMinimum    = 80.0;
    std::string              llvm            = "-21";
    std::vector<std::string> hotPathBinaries = {"core/src/main"};
    std::vector<std::string> wasmModules     = {"core/src/main.wasm"};
    double                   hotPathLimit    = 3.0; // % rise named in the commit message (CLAUDE.md, measured quality)
    double                   hotPathNoise    = 0.1; // % below which a change reads ±0; the listed benchmarks repeat within 0.015 % under callgrind (x86-64-v3, 2026-10-16)
};
const Settings kSettings;

constexpr std::string_view kHistoryReport = "history-report.txt";

std::string settingText(std::string_view name) {
    const auto joined = [](const std::vector<std::string>& values) {
        std::string text;
        for (const std::string& value : values) {
            text += value + '\n';
        }
        return text;
    };
    if (name == "namingArguments") {
        return joined(kSettings.namingArguments);
    }
    if (name == "coverageAreas") {
        std::string text;
        for (const auto& [area, prefix] : kSettings.coverageAreas) {
            text += area + '=' + prefix + '\n';
        }
        return text;
    }
    if (name == "coverageExcluded") {
        return kSettings.coverageExcluded;
    }
    if (name == "coverageMethod") {
        return kSettings.coverageMethod;
    }
    if (name == "hotPathBinaries") {
        return joined(kSettings.hotPathBinaries);
    }
    if (name == "wasmModules") {
        return joined(kSettings.wasmModules);
    }
    return {};
}

struct BotInfo {
    std::string_view              title;
    std::string_view              context;
    std::vector<std::string_view> tool; // files, settings, `version:<command>`, `env:<variable>`
};

const std::map<std::string_view, BotInfo> kBots = {
    {"formatting", {"Formatting", "formatting/files", {"formatFiles.sh", ".clang-format", ".cmake-format.yaml"}}},
    {"naming", {"Naming", "naming/violations", {"version:clang-22", "devtools/naming-check.cpp", "devtools/NameSpelling.hpp", "devtools/Libclang.hpp", "namingArguments"}}},
    {"code-style", {"Code Style", "code-style/compare", {"version:clang-22", "version:clang-tidy-22", "devtools/code-metrics.cpp", "devtools/code-metrics.sh", "devtools/Figures.hpp", "devtools/Ccd.hpp", "devtools/Libclang.hpp"}}},
    {"coverage", {"Coverage", "coverage/patch", {"version:acpp", "version:llvm-cov-21", "coverageAreas", "coverageExcluded", "coverageMethod"}}},
    {"hot-paths", {"Hot Paths", "hot-paths/instructions", {"version:g++-15", "version:valgrind", "devtools/ci/hot-path-benchmarks.txt", "hotPathBinaries"}}},
    {"wasm-size", {"WASM size", "hot-paths/wasm-size", {"env:EMSDK_VERSION", "wasmModules"}}},
};

// the tools beside this binary's sources measure any checked-out tree
const fs::path kRepository = fs::path{__FILE__}.parent_path().parent_path();

struct Ran {
    int         status = 0;
    std::string out;
    std::string err;
};

std::string readFile(const fs::path& path) {
    std::ifstream     file{path, std::ios::binary};
    std::stringstream text;
    text << file.rdbuf();
    return text.str();
}

void writeFile(const fs::path& path, std::string_view text) {
    std::ofstream file{path, std::ios::binary};
    file << text;
}

std::string shellQuoted(std::string_view word) {
    std::string quoted = "'";
    for (const char character : word) {
        quoted += character == '\'' ? std::string{"'\\''"} : std::string(1, character);
    }
    return quoted + "'";
}

Ran run(const std::vector<std::string>& command, std::string_view input = {}, const std::vector<std::string>& environment = {}) {
    static std::atomic<unsigned> counter{0U};
    const fs::path               scratch = fs::temp_directory_path() / std::format("qualitybot-{}-{}", ::getpid(), counter++);
    fs::create_directories(scratch);
    writeFile(scratch / "in", input);
    std::string line;
    for (const std::string& assignment : environment) {
        line += assignment.substr(0, assignment.find('=') + 1) + shellQuoted(assignment.substr(assignment.find('=') + 1)) + ' ';
    }
    for (const std::string& word : command) {
        line += shellQuoted(word) + ' ';
    }
    line += std::format("<{} >{} 2>{}", shellQuoted((scratch / "in").string()), shellQuoted((scratch / "out").string()), shellQuoted((scratch / "err").string()));
    const int raw = std::system(line.c_str());
    Ran       ran{.status = WIFEXITED(raw) ? WEXITSTATUS(raw) : 128, .out = readFile(scratch / "out"), .err = readFile(scratch / "err")};
    fs::remove_all(scratch);
    return ran;
}

// as `run`, but a failure other than "found something" (exit 1) ends the bot
Ran checked(const std::vector<std::string>& command, std::string_view input = {}, const std::vector<std::string>& environment = {}) {
    Ran ran = run(command, input, environment);
    if (ran.status != 0 && ran.status != 1) {
        std::println(stderr, "{} failed ({}):\n{}", joined(command, " "), ran.status, ran.err);
        std::exit(2);
    }
    return ran;
}

std::string environmentOr(const char* name, std::string_view fallback) {
    const char* value = std::getenv(name);
    return value != nullptr && *value != '\0' ? std::string{value} : std::string{fallback};
}

std::vector<std::string> wordsOf(std::string_view text) {
    std::istringstream stream{std::string{text}};
    return {std::istream_iterator<std::string>{stream}, std::istream_iterator<std::string>{}};
}

std::string marker(std::string_view bot) { return std::format("<!-- {}:{} -->", kSettings.marker, bot); }

std::vector<Row> history(std::string_view bot) {
    const Ran shown = run({"git", "show", std::format("{}:{}/history.csv", kSettings.historyRef, bot)});
    return shown.status == 0 ? csvTable(shown.out).rows : std::vector<Row>{};
}

// a hash of the bot's tools: a change of method breaks the comparison, a squash merge or an unrelated image edit does not
std::string toolVersion(std::string_view bot) {
    constexpr std::string_view kCommand  = "version:";
    constexpr std::string_view kVariable = "env:";
    std::string                content;
    for (const std::string_view name : kBots.at(bot).tool) {
        if (name.starts_with(kCommand)) {
            const Ran         reported = run({std::string{name.substr(kCommand.size())}, "--version"});
            const std::string version  = reported.status == 0 ? reported.out : std::string{"absent"};
            content += version.substr(0, version.find('\n'));
        } else if (name.starts_with(kVariable)) {
            content += environmentOr(std::string{name.substr(kVariable.size())}.c_str(), "absent");
        } else {
            const std::string setting = settingText(name);
            content += setting.empty() ? readFile(kRepository / name) : setting;
        }
    }
    return trimmed(checked({"git", "hash-object", "--stdin"}, content).out).substr(0, 10);
}

// the main commit this change lands on: a pull request's checkout is its test merge, whose first parent is main; a
// main commit's base is its parent; elsewhere the merge-base with origin/main
Base baseline(std::string_view bot) {
    const std::vector<std::string> shas      = wordsOf(checked({"git", "rev-list", "--parents", "-n", "1", "HEAD"}).out);
    const std::string              head      = shas.empty() ? std::string{} : shas.front();
    const std::string              mergeBase = trimmed(run({"git", "merge-base", "HEAD", "origin/main"}).out);
    const bool                     onMain    = shas.size() > 2U || mergeBase.empty() || mergeBase == head;
    const std::string              commit    = shas.size() > 1U && onMain ? shas[1] : mergeBase;
    const std::vector<std::string> ancestors = commit.empty() ? std::vector<std::string>{} : wordsOf(run({"git", "rev-list", "--first-parent", "--max-count=500", commit}).out);
    return baseFrom(commit, ancestors, history(bot), toolVersion(bot));
}

struct Links {
    std::string head;
    std::string run;
    std::string pages;
};

// head: a pull request's own commit, not its test merge
Links links() {
    const std::vector<std::string> shas        = wordsOf(run({"git", "rev-list", "--parents", "-n", "1", "HEAD"}).out);
    const bool                     pullRequest = environmentOr("GITHUB_EVENT_NAME", "") == "pull_request";
    const std::string              head        = pullRequest && shas.size() == 3U ? shas[2] : environmentOr("GITHUB_SHA", shas.empty() ? std::string{} : shas.front());
    const std::string repository = environmentOr("GITHUB_REPOSITORY", "");
    const std::string              server      = environmentOr("GITHUB_SERVER_URL", "https://github.com");
    const std::string              runId       = environmentOr("GITHUB_RUN_ID", "");
    const std::size_t              slash       = repository.find('/');
    std::string                    owner       = repository.substr(0, slash);
    std::ranges::transform(owner, owner.begin(), [](char character) { return static_cast<char>(std::tolower(static_cast<unsigned char>(character))); });
    return {.head = head, .run = !repository.empty() && !runId.empty() ? std::format("{}/{}/actions/runs/{}", server, repository, runId) : std::string{}, .pages = slash != std::string::npos ? std::format("https://{}.github.io/{}", owner, repository.substr(slash + 1)) : std::string{}};
}

std::string trend(std::string_view bot, std::string_view title, const Base& base, std::string_view figure, std::optional<double> current, std::string_view unit) {
    constexpr long long kSecondsPerDay = 86'400;
    const std::string   headDate       = trimmed(run({"git", "log", "-1", "--format=%cI", links().head}).out);
    return chart(title, chartPoints(base.rows, figure, current, headDate, toolVersion(bot), kSettings.chartPoints, kSettings.chartMaxAgeDays * kSecondsPerDay), unit);
}

using Figures = std::vector<std::pair<std::string, std::optional<double>>>;

struct Report {
    std::string_view         dot;
    std::string              headline;
    std::string              reason;
    std::string              body;
    std::string              legend;
    Figures                  figures;
    std::vector<std::string> footerLinks = {};
};

void writeReport(const fs::path& out, std::string_view bot, const Base& base, const Report& report) {
    const Links              where  = links();
    const std::string        onto   = base.commit.empty() ? std::string{} : std::format(" → main @ `{}`", base.commit.substr(0, 8));
    const std::string        note   = base.note.empty() ? std::string{} : std::format(" ({})", base.note);
    const std::string        reason = report.reason.empty() ? std::string{} : std::format(" · {}", report.reason);
    const std::string        header = std::format("{}\n### {}{}{} · {}\n`{}`{}{}{}\n\n", marker(bot), report.dot, report.dot.empty() ? "" : " ", kBots.at(bot).title, report.headline, where.head.substr(0, 8), onto, note, reason);
    std::vector<std::string> footer{report.legend};
    footer.insert(footer.end(), report.footerLinks.begin(), report.footerLinks.end());
    if (!where.run.empty()) {
        footer.push_back(std::format("[run and artefacts]({})", where.run));
    }
    if (!where.pages.empty()) {
        footer.push_back(std::format("[history]({}/metrics/)", where.pages));
    }
    const auto        isShown     = [](const std::string& part) { return !part.empty(); };
    const std::string comment     = header + trimmed(report.body) + "\n\n" + joined(footer | std::views::filter(isShown), " · ") + "\n";
    const std::string description = report.reason.empty() ? report.headline : std::format("{}; {}", report.reason, report.headline);
    const std::string state       = report.dot == kRed ? "failure" : "success";
    writeFile(out / "comment.md", comment);
    writeFile(out / "status.json", std::format("{{\"context\": {}, \"state\": {}, \"description\": {}}}", jsonQuoted(kBots.at(bot).context), jsonQuoted(state), jsonQuoted(cutUtf8(description, 140U))));
    Table table{.columns = {"bot", "tool"}, .rows = {Row{{"bot", std::string{bot}}, {"tool", toolVersion(bot)}}}};
    for (const auto& [name, value] : report.figures) {
        table.columns.push_back(name);
        table.rows.front()[name] = value ? shortest(*value) : std::string{};
    }
    writeFile(out / "figures.csv", csvText(table));
    std::print("{}", comment);
}

std::string valueOf(const Row& row, std::string_view column) {
    const auto found = row.find(column);
    return found == row.end() ? std::string{} : found->second;
}

std::string formatterOf(std::string_view name) {
    for (const auto& [tool, endings] : kSettings.formatters) {
        if (std::ranges::any_of(endings, [name](const std::string& ending) { return name.ends_with(ending); })) {
            return tool;
        }
    }
    return "whitespace";
}

struct Arguments {
    std::string           bot;
    fs::path              out;
    fs::path              build;
    fs::path              profiles;
    std::vector<fs::path> reportDirectories;
};

void formatting(const Arguments& arguments) {
    const Base                  base    = baseline("formatting");
    const std::set<std::string> touched = base.commit.empty() ? std::set<std::string>{} : [&] {
        const std::vector<std::string> names = linesOf(checked({"git", "diff", "--name-only", "--diff-filter=d", base.commit}).out);
        return std::set<std::string>{names.begin(), names.end()};
    }();
    // a file counts when the formatters change it, so a local run on a tree with edits reports only what they did
    const auto modified = [] {
        std::map<std::string, std::string> contents;
        for (const std::string& name : linesOf(checked({"git", "diff", "--name-only"}).out)) {
            if (fs::is_regular_file(name)) {
                contents[name] = readFile(name);
            }
        }
        return contents;
    };
    const auto before    = modified();
    const Ran  formatted = run({"./formatFiles.sh", "--all"});
    const auto after     = modified();
    writeFile(arguments.out / "formatFiles.log", formatted.out + formatted.err);
    const fs::path        scratch = arguments.out / "before";
    std::set<std::string> names;
    for (const auto& [name, _] : before) {
        names.insert(name);
    }
    for (const auto& [name, _] : after) {
        names.insert(name);
    }
    std::vector<std::string> files;
    std::string              patch;
    for (const std::string& name : names) {
        const std::string contentBefore = before.contains(name) ? before.at(name) : checked({"git", "show", std::format("HEAD:{}", name)}).out;
        const std::string contentAfter  = after.contains(name) ? after.at(name) : checked({"git", "show", std::format("HEAD:{}", name)}).out;
        if (contentBefore == contentAfter) {
            continue;
        }
        files.push_back(name);
        fs::create_directories((scratch / name).parent_path());
        writeFile(scratch / name, contentBefore);
        patch += run({"diff", "-u", "--label", "a/" + name, "--label", "b/" + name, (scratch / name).string(), name}).out;
    }
    fs::remove_all(scratch);
    writeFile(arguments.out / "formatting.patch", patch);
    std::vector<std::string> mine;
    std::vector<std::string> elsewhere;
    for (const std::string& name : files) {
        (touched.contains(name) ? mine : elsewhere).push_back(name);
    }
    const auto table = [](const std::vector<std::string>& selected) {
        std::string text = "| file | formatter |\n| --- | --- |\n";
        for (const std::string& name : selected) {
            text += std::format("| `{}` | {} |\n", name, formatterOf(name));
        }
        return text + "\n";
    };
    std::string body;
    if (!mine.empty()) {
        body += "Run `./formatFiles.sh --since origin/main` and commit the result; the full patch is this run's `formatting-patch` artefact.\n\n" + table(mine);
    }
    body += trend("formatting", "files left unformatted", base, "files", static_cast<double>(files.size()), "files");
    if (!elsewhere.empty()) {
        body += details(std::format("{} this change does not touch, not counted", plural(elsewhere.size(), "file")), table(std::vector<std::string>(elsewhere.begin(), elsewhere.begin() + static_cast<std::ptrdiff_t>(std::min(elsewhere.size(), kSettings.tableRows)))));
    }
    if (!files.empty()) {
        const std::vector<std::string> diffLines = linesOf(patch);
        std::string                    shown;
        for (const std::string& line : diffLines | std::views::take(kSettings.diffLines)) {
            shown += line + '\n';
        }
        const std::string more = diffLines.size() > kSettings.diffLines ? std::format("\n\n… {} more lines in the artefact", diffLines.size() - kSettings.diffLines) : std::string{};
        body += details("The changes the formatters make", fenced(shown, "diff") + more);
    }
    body += details("What is checked", "`./formatFiles.sh` runs Restyled's formatters (clang-format, cmake-format, prettier for Markdown and YAML, shellcheck's fixes, whitespace, black) at the versions it pins, with the repository's configuration.\n\n" + fenced(trimmed(run({"./formatFiles.sh", "--versions"}).out)));
    const std::string_view dot      = mine.empty() ? kGreen : kRed;
    const std::string      headline = std::format("{} of {} changed files need formatting ({} in all)", mine.size(), touched.size(), files.size());
    const std::string      reason   = mine.empty() ? "" : "run `./formatFiles.sh --since origin/main`";
    const std::string      legend   = std::format("{} the changed files are formatted · {} a changed file needs formatting", kGreen, kRed);
    const Figures          figures{{"files", static_cast<double>(files.size())}};
    writeReport(arguments.out, "formatting", base, {.dot = dot, .headline = headline, .reason = reason, .body = body, .legend = legend, .figures = figures});
}

struct Violation {
    std::string where;
    std::string file;
    unsigned    line = 0;
    std::string kind;
    std::string name;
    std::string rule;
    std::string proposal;
};

void naming(const Arguments& arguments) {
    const Base               base = baseline("naming");
    const std::string        tool = (arguments.build / "devtools" / "naming-check").string();
    std::vector<std::string> command{tool, arguments.build.string()};
    command.insert(command.end(), kSettings.namingArguments.begin(), kSettings.namingArguments.end());
    command.insert(command.end(), {"--jobs", std::to_string(kSettings.parseJobs), "--list"});
    const std::vector<std::string> lines = linesOf(checked(command).out);
    const std::regex               countPattern{R"(^translation units checked: (\d+) of (\d+)$)"};
    std::smatch                    counted;
    const auto                     countLine = std::ranges::find_if(lines, [&](const std::string& line) { return std::regex_match(line, counted, countPattern); });
    if (countLine == lines.end()) {
        std::println(stderr, "naming-check gave no unit count");
        std::exit(2);
    }
    const std::string      checkedUnits = counted[1].str();
    const std::string      units        = counted[2].str();
    const std::regex       violationPattern{R"(^  (\S+) (.+?) '([^']*)' \((R\d+): (.*)\)(?: -> (\S+))?$)"};
    std::vector<Violation> violations;
    const auto             listed = std::ranges::find(lines, std::string{"violations"});
    for (const std::string& line : std::ranges::subrange(listed == lines.end() ? lines.end() : listed + 1, lines.end())) {
        std::smatch found;
        if (!std::regex_match(line, found, violationPattern)) {
            continue;
        }
        const std::string where  = found[1].str();
        const std::size_t colon  = where.find(':');
        unsigned          number = 0;
        if (colon != std::string::npos) {
            std::from_chars(where.data() + colon + 1, where.data() + where.size(), number);
        }
        violations.push_back({.where = where, .file = where.substr(0, colon), .line = number, .kind = found[2].str(), .name = found[3].str(), .rule = found[4].str(), .proposal = found[6].matched ? std::format("`{}`", found[6].str()) : "—"});
    }
    const auto changed = base.commit.empty() ? std::map<std::string, std::set<unsigned>, std::less<>>{} : changedLinesIn(run({"git", "diff", "-U0", "--no-color", base.commit + "...HEAD"}).out);
    // a file's or directory's name (line 0) is this change's when the base has no such path
    const auto                         isNew = [&base](const std::string& path) { return !base.commit.empty() && run({"git", "cat-file", "-e", std::format("{}:{}", base.commit, path)}).status != 0; };
    std::vector<Violation>             mine;
    std::map<std::string, std::size_t> perRule;
    for (const Violation& violation : violations) {
        ++perRule[violation.rule];
        const auto changedInFile = changed.find(violation.file);
        if (violation.line == 0U ? isNew(violation.file) : changedInFile != changed.end() && changedInFile->second.contains(violation.line)) {
            mine.push_back(violation);
        }
    }
    const auto table = [](const std::vector<Violation>& selected) {
        std::string text = "| where | rule | kind | name | proposed |\n| --- | --- | --- | --- | --- |\n";
        for (const Violation& v : selected | std::views::take(kSettings.tableRows)) {
            text += std::format("| `{}` | {} | {} | `{}` | {} |\n", v.where, v.rule, v.kind, v.name, v.proposal);
        }
        return text;
    };
    std::string body = mine.empty() ? std::string{} : table(mine) + "\n";
    body += trend("naming", "naming violations", base, "violations", static_cast<double>(violations.size()), "violations");
    const std::string        rulesText = checked({tool, "--rules"}).out;
    std::vector<std::string> rules;
    for (const std::string& line : linesOf(rulesText)) {
        std::smatch rule;
        if (std::regex_search(line, rule, std::regex{R"(^(R\d+) )"})) {
            rules.push_back(rule[1].str());
        }
    }
    if (!violations.empty()) {
        const std::string more = violations.size() > kSettings.tableRows ? std::format("\n… {} more: `naming-check --list`", violations.size() - kSettings.tableRows) : std::string{};
        body += details(std::format("All {}", plural(violations.size(), "violation")), table(violations) + more);
        std::string byRule;
        for (const std::string& rule : rules) {
            if (perRule.contains(rule)) {
                byRule += std::format("- {}: {}\n", rule, perRule.at(rule));
            }
        }
        body += details("By rule", byRule);
    }
    body += details("The rules (CLAUDE.md §3)", fenced(rulesText));
    const std::string namingArguments = joined(kSettings.namingArguments, " ");
    body += details("Method", std::format("`naming-check {}` on the Clang build's compile database; {} of {} units checked; this change's lines: since {}.", namingArguments, checkedUnits, units, base.commit.empty() ? "—" : base.commit.substr(0, 8)));
    Figures figures{{"violations", static_cast<double>(violations.size())}, {"units", numberIn(units)}};
    for (const std::string& rule : rules) {
        figures.emplace_back(rule, static_cast<double>(perRule.contains(rule) ? perRule.at(rule) : 0U));
    }
    const std::string_view dot      = mine.empty() ? kGreen : kRed;
    const std::string      headline = std::format("{} on this change's lines ({} in all)", plural(mine.size(), "violation"), violations.size());
    const std::string      reason   = mine.empty() ? "" : "rename them";
    const std::string      legend   = std::format("{} no violation on a changed line · {} violations on changed lines", kGreen, kRed);
    writeReport(arguments.out, "naming", base, {.dot = dot, .headline = headline, .reason = reason, .body = body, .legend = legend, .figures = figures});
}

struct Compared {
    std::string              group;
    std::string              name;
    std::string              before;
    std::string              after;
    std::string              change;
    std::string              verdict;
    std::vector<std::string> places;
};

// `(g) name before -> after change verdict` lines of a --compare report, each with the indented lines beneath it
std::vector<Compared> compared(std::string_view text) {
    std::vector<Compared> rows;
    const std::regex      pattern{R"(^\((\d)\) (.+?)\s{2,}(\S+(?: %)?|—)\s+->\s+(\S+(?: %)?)\s+(\S+ %|\+\d+|-\d+)?\s*(\S.*)$)"};
    for (const std::string& line : linesOf(text)) {
        std::smatch found;
        if (std::regex_match(line, found, pattern)) {
            rows.push_back({.group = found[1].str(), .name = trimmed(found[2].str()), .before = found[3].str(), .after = found[4].str(), .change = found[5].matched ? found[5].str() : "", .verdict = trimmed(found[6].str()), .places = {}});
        } else if (!rows.empty() && line.starts_with("      ")) {
            rows.back().places.push_back(trimmed(line));
        }
    }
    return rows;
}

void codeStyle(const Arguments& arguments) {
    const Base                     base = baseline("code-style");
    const std::vector<std::string> script{(kRepository / "devtools" / "code-metrics.sh").string(), arguments.build.string(), "--cognitive", "--jobs", std::to_string(kSettings.parseJobs)};
    const auto                     withScript = [&script](std::vector<std::string> extra) {
        std::vector<std::string> command = script;
        command.insert(command.end(), extra.begin(), extra.end());
        return command;
    };
    const std::string report = checked(withScript({"--list"})).out;
    std::smatch       unparsed;
    const std::size_t unparsedCount = std::regex_search(report, unparsed, std::regex{R"(translation units with parse errors \(their figures are incomplete\): (\d+))"}) ? std::stoul(unparsed[1].str()) : 0U;
    writeFile(arguments.out / kHistoryReport, report);
    const auto current = figureLines(report);
    // a tool that did not run leaves no figures, or no cognitive figure: a report of them would enter main's history
    if (current.empty() || std::ranges::find(current, kSettings.codeStyleChart, &std::pair<std::string, std::string>::first) == current.end()) {
        std::println(stderr, "code-metrics.sh gave no '{}' figure:\n{}", kSettings.codeStyleChart, report);
        std::exit(2);
    }
    const auto currentOf = [&current](std::string_view name) {
        const auto found = std::ranges::find(current, name, &std::pair<std::string, std::string>::first);
        return found == current.end() ? std::string{"—"} : found->second;
    };
    const std::string key = kSettings.codeStyleChart;
    std::string       body;
    // main's report of the same commit as its figures, so what moved a figure is this change's
    const Ran   baseReport = base.row.empty() ? Ran{.status = 1, .out = {}, .err = {}} : run({"git", "show", std::format("{}:code-style/reports/{}.txt", kSettings.historyRef, valueOf(base.row, "commit"))});
    const bool  comparable = baseReport.status == 0;
    std::size_t worse      = 0;
    std::size_t better     = 0;
    std::string headline;
    if (!comparable) {
        body += trend("code-style", key, base, key, numberIn(currentOf(key)), key);
        std::string all = "| figure | value |\n| --- | ---: |\n";
        for (const auto& [name, value] : current) {
            all += std::format("| {} | {} |\n", name, value);
        }
        body += details("All figures", all);
        headline = std::format("{} figures measured ({} {})", current.size(), key, currentOf(key));
    } else {
        writeFile(arguments.out / "base-list.txt", baseReport.out);
        const std::vector<Compared> comparison = compared(checked(withScript({"--compare", (arguments.out / "base-list.txt").string(), "--list"})).out);
        const auto                  isCounted  = [](const Compared& row) { return std::ranges::contains(kSettings.codeStyleCounted, row.name); };
        std::vector<Compared>       moved;
        for (const Compared& row : comparison) {
            if (row.verdict == "worse" || row.verdict == "better") {
                moved.push_back(row);
            }
            worse += isCounted(row) && row.verdict == "worse" ? 1U : 0U;
            better += isCounted(row) && row.verdict == "better" ? 1U : 0U;
        }
        std::ranges::stable_sort(moved, {}, [&isCounted](const Compared& row) { return std::pair{!isCounted(row), row.verdict != "worse"}; });
        const auto dotOf = [&isCounted](const Compared& row) { return isCounted(row) ? (row.verdict == "better" ? kGreen : kYellow) : std::string_view{}; };
        const auto delta = [](const Compared& row) {
            const std::optional<double> before = numberIn(row.before);
            const std::optional<double> after  = numberIn(row.after);
            if (!before || !after) {
                return row.change;
            }
            const bool        percent  = row.after.ends_with("%");
            const std::string absolute = percent ? std::format("{:+.1f} pp", *after - *before) : std::format("{}{}", *after - *before >= 0.0 ? "+" : "", std::abs(*after - *before - std::round(*after - *before)) < 1e-9 ? withThousands(std::llround(*after - *before)) : shortest(*after - *before));
            return row.change.ends_with("%") && !percent ? std::format("{} ({})", absolute, row.change) : absolute;
        };
        if (!moved.empty()) {
            body += "| | figure | Δ | now |\n| --- | --- | ---: | ---: |\n";
            for (const Compared& row : moved) {
                body += std::format("| {} | {}{} | {} | {} |\n", dotOf(row), row.name, isCounted(row) ? "" : " *(not counted)*", delta(row), row.after);
            }
            body += "\n";
        }
        body += trend("code-style", key, base, key, numberIn(currentOf(key)), key);
        for (const Compared& row : moved) {
            if (!row.places.empty()) {
                body += details(std::format("{} {}: what moved it", dotOf(row), row.name), fenced(joined(row.places, "\n")));
            }
        }
        std::string all = "| figure | main | this | change | |\n| --- | ---: | ---: | ---: | --- |\n";
        for (const Compared& row : comparison) {
            all += std::format("| ({}) {} | {} | {} | {} | {} |\n", row.group, row.name, row.before, row.after, row.change, row.verdict);
        }
        body += details("All figures", all);
        headline = std::format("{} ({} {})", worse != 0U || better != 0U ? std::format("{} worse, {} better", plural(worse, "figure"), better) : std::string{"no counted figure moved"}, key, currentOf(key));
    }
    const std::string counted = joined(kSettings.codeStyleCounted, ", ");
    body += details("Counted figures", counted + ": the sums CLAUDE.md (measured quality) names. Ratios and the other figures are shown, not counted: a deletion alone moves them.");
    body += details("How to read the figures", fenced(run({(kRepository / "devtools" / "code-metrics.sh").string(), "--figures"}).out) + "\n\nEach figure's card: `devtools/code-metrics.sh --help`.");
    std::string_view dot    = worse != 0U ? kYellow : comparable ? kGreen : std::string_view{};
    std::string      reason = worse != 0U ? "name the worse ones in the commit message (CLAUDE.md)" : "";
    if (unparsedCount != 0U) {
        // figures over a partly parsed tree are wrong, not merely incomplete: say so first and turn red
        body   = std::format("> [!CAUTION]\n> {} could not be parsed, so the figures are incomplete.\n\n", plural(unparsedCount, "translation unit")) + body;
        dot    = kRed;
        reason = std::format("{} not parsed", plural(unparsedCount, "unit"));
    }
    Figures figures;
    for (const auto& [name, value] : current) {
        figures.emplace_back(name, numberIn(value));
    }
    const std::string legend = std::format("{} no counted figure worse · {} a counted figure worse: name it in the commit message · {} units not parsed", kGreen, kYellow, kRed);
    writeReport(arguments.out, "code-style", base, {.dot = dot, .headline = headline, .reason = reason, .body = body, .legend = legend, .figures = figures});
}

template<typename TTask>
void inParallel(std::size_t count, TTask&& task) {
    std::atomic<std::size_t>  next{0U};
    std::vector<std::jthread> workers;
    for (unsigned k = 0; k < std::max(1U, std::thread::hardware_concurrency()); ++k) {
        workers.emplace_back([&] {
            for (std::size_t index = next++; index < count; index = next++) {
                task(index);
            }
        });
    }
}

bool isOwnBuildFile(const fs::path& path) {
    return std::ranges::none_of(path, [](const fs::path& part) { return part == "_deps" || part == "CMakeFiles"; });
}

// the build's executables and shared libraries that carry a coverage mapping llvm-cov reads, and those it rejects:
// Clang 20 emits a mapping for some units that llvm-cov refuses, and one such object would abort the whole report
std::pair<std::vector<std::string>, std::vector<std::string>> instrumented(const fs::path& build, const fs::path& merged) {
    std::vector<fs::path> candidates;
    for (const auto& entry : fs::recursive_directory_iterator(build)) {
        if (entry.is_regular_file() && (entry.status().permissions() & fs::perms::owner_exec) != fs::perms::none && isOwnBuildFile(entry.path())) {
            candidates.push_back(entry.path());
        }
    }
    std::ranges::sort(candidates);
    enum class Mapping { none, readable, rejected };
    std::vector<Mapping> verdicts(candidates.size(), Mapping::none);
    inParallel(candidates.size(), [&](std::size_t k) {
        if (!run({"llvm-objdump" + kSettings.llvm, "--section-headers", candidates[k].string()}).out.contains("__llvm_covmap")) {
            return;
        }
        const Ran report = run({"llvm-cov" + kSettings.llvm, "report", "-instr-profile=" + merged.string(), candidates[k].string()});
        if (report.status == 0) {
            verdicts[k] = Mapping::readable;
        } else if (!report.err.contains("no coverage data found")) {
            verdicts[k] = Mapping::rejected;
        }
    });
    std::pair<std::vector<std::string>, std::vector<std::string>> objects;
    for (std::size_t k = 0; k < candidates.size(); ++k) {
        if (verdicts[k] == Mapping::readable) {
            objects.first.push_back(candidates[k].string());
        } else if (verdicts[k] == Mapping::rejected) {
            objects.second.push_back(candidates[k].string());
        }
    }
    return objects;
}

std::map<std::string, std::map<unsigned, unsigned long long>, std::less<>> lcovLines(std::string_view lcov, const fs::path& root) {
    std::map<std::string, std::map<unsigned, unsigned long long>, std::less<>> lines;
    std::string                                                                current;
    for (const std::string& line : linesOf(lcov)) {
        if (line.starts_with("SF:")) {
            const fs::path path{line.substr(3)};
            const auto     relative = path.lexically_relative(root).string();
            current                 = path.is_absolute() && !relative.starts_with("..") ? relative : path.string();
            lines[current];
        } else if (line.starts_with("DA:") && !current.empty()) {
            unsigned           number = 0;
            unsigned long long hits   = 0;
            const char*        end    = line.data() + line.size();
            const auto         parsed = std::from_chars(line.data() + 3, end, number);
            std::from_chars(parsed.ptr + 1, end, hits);
            lines[current][number] += hits;
        }
    }
    return lines;
}

std::string listedLines(const std::map<std::string, std::set<unsigned>, std::less<>>& byFile, std::size_t cap = 20) {
    std::vector<std::string> entries;
    for (const auto& [name, numbers] : byFile) {
        if (!numbers.empty()) {
            entries.push_back(std::format("- `{}` {}\n", name, rangesOf(numbers)));
        }
    }
    std::string text;
    for (const std::string& entry : entries | std::views::take(cap)) {
        text += entry;
    }
    return entries.size() > cap ? text + std::format("- + {} more files, all in the artefact\n", entries.size() - cap) : text;
}

// a line without coverage data that is no blank, comment, preprocessor, brace or declaration line: mostly an uninstantiated template
bool executableLooking(std::string_view raw) {
    const std::string text = trimmed(raw);
    if (text.empty() || text.starts_with("//") || text.starts_with("/*") || text.starts_with("*") || text.starts_with("#") || std::regex_match(text, std::regex{R"([{}();,]*)"}) || std::regex_search(text, std::regex{R"(^\}\s*//)"}) || text == "public:" || text == "private:" || text == "protected:") {
        return false;
    }
    return !std::regex_search(text, std::regex{R"(^(template\s*<|using |struct |class |enum |namespace |friend |static_assert|requires|concept |typename |(gr::)?(Annotated|A)<|(gr::)?(PortIn|PortOut|MsgPortIn|MsgPortOut)<|GR_MAKE_REFLECTABLE|GR_REGISTER_BLOCK))"});
}

// `llvm-cov report`'s TOTAL row: per group its count, missed count and share
std::map<std::string, std::optional<double>> reportTotals(const std::vector<std::string>& command) {
    const std::vector<std::string> report = linesOf(checked(command).out);
    const auto                     header = std::ranges::find_if(report, [](const std::string& line) { return line.starts_with("Filename"); });
    const auto                     total  = std::ranges::find_if(report, [](const std::string& line) { return line.starts_with("TOTAL"); });
    std::vector<std::string>       names;
    if (header != report.end()) {
        const std::string          trimmedHeader = trimmed(*header);
        const std::regex           gap{R"(\s{2,})"};
        std::sregex_token_iterator split{trimmedHeader.begin(), trimmedHeader.end(), gap, -1};
        names.assign(std::next(split), std::sregex_token_iterator{});
    }
    std::vector<std::string> values = total == report.end() ? std::vector<std::string>{} : wordsOf(*total);
    if (!values.empty()) {
        values.erase(values.begin());
    }
    std::map<std::string, std::optional<double>> totals;
    for (const auto& [key, group] : {std::pair{"regions", "Regions"}, {"functions", "Functions"}, {"instantiations", "Instantiations"}, {"lines", "Lines"}, {"branches", "Branches"}}) {
        const auto        found = std::ranges::find(names, std::string{group});
        const std::size_t index = static_cast<std::size_t>(found - names.begin()) + 2U;
        const auto        share = found != names.end() && index < values.size() ? numberIn(values[index]) : std::nullopt;
        totals[key]             = share ? std::optional<double>{std::round(*share * 10.0) / 10.0} : std::nullopt;
    }
    return totals;
}

std::optional<double> percent(std::size_t found, std::size_t hit) { return found == 0U ? std::nullopt : std::optional<double>{std::round(1000.0 * static_cast<double>(hit) / static_cast<double>(found)) / 10.0}; }

std::string shownPercent(std::optional<double> share) { return share ? std::format("{:.1f}", *share) : "None"; }

void coverage(const Arguments& arguments) {
    const Base               base = baseline("coverage");
    const fs::path           root = fs::current_path();
    std::vector<std::string> profiles;
    for (const auto& entry : fs::directory_iterator(arguments.profiles)) {
        if (entry.path().extension() == ".profraw") {
            profiles.push_back(entry.path().string());
        }
    }
    std::ranges::sort(profiles);
    if (profiles.empty()) {
        std::println(stderr, "no profiles in {}", arguments.profiles.string());
        std::exit(2);
    }
    const fs::path           merged = arguments.out / "coverage.profdata";
    std::vector<std::string> merge{"llvm-profdata" + kSettings.llvm, "merge", "-sparse", "-failure-mode=all"};
    merge.insert(merge.end(), profiles.begin(), profiles.end());
    merge.insert(merge.end(), {"-o", merged.string()});
    checked(merge);
    const auto [objects, rejected] = instrumented(arguments.build, merged);
    writeFile(arguments.out / "coverage-skipped.txt", joined(rejected, "\n"));
    if (objects.empty()) {
        std::println(stderr, "no binary with a readable coverage mapping in {}", arguments.build.string());
        std::exit(2);
    }
    const std::string        llvmCov = "llvm-cov" + kSettings.llvm;
    std::vector<std::string> selection{"-instr-profile=" + merged.string(), "-ignore-filename-regex=" + kSettings.coverageExcluded, objects.front()};
    for (const std::string& path : objects | std::views::drop(1)) {
        selection.insert(selection.end(), {"-object", path});
    }
    const auto llvmCovWith = [&](std::vector<std::string> verb) {
        verb.insert(verb.begin(), llvmCov);
        verb.insert(verb.end(), selection.begin(), selection.end());
        return verb;
    };
    // one export per binary, summed by lcovLines: a joint export keeps one function record per name and drops the counts of the others
    std::vector<std::string> lcovPerObject(objects.size());
    inParallel(objects.size(), [&](std::size_t k) { lcovPerObject[k] = checked({llvmCov, "export", "-format=lcov", "-instr-profile=" + merged.string(), "-ignore-filename-regex=" + kSettings.coverageExcluded, objects[k]}).out; });
    const std::string lcov = joined(lcovPerObject, "");
    writeFile(arguments.out / "coverage.lcov", lcov);
    checked(llvmCovWith({"show", "-format=html", "-output-dir=" + (arguments.out / "html").string(), "-show-instantiations=false"}));
    // Sonar reads llvm-cov's text report (sonar.cfamily.llvm-cov.reportPath)
    writeFile(arguments.out / "coverage.txt", checked(llvmCovWith({"show", "-show-instantiations=false"})).out);
    const auto totals = reportTotals(llvmCovWith({"report", "-show-instantiation-summary"}));
    const auto lines  = lcovLines(lcov, root);
    const auto inArea = [](std::string_view name) { return std::ranges::any_of(kSettings.coverageAreas, [name](const auto& area) { return name.starts_with(area.second); }); };
    const auto tally  = [&lines](auto&& wanted) {
        std::size_t found = 0;
        std::size_t hit   = 0;
        for (const auto& [name, hits] : lines) {
            if (wanted(name)) {
                found += hits.size();
                hit += static_cast<std::size_t>(std::ranges::count_if(hits, [](const auto& entry) { return entry.second > 0U; }));
            }
        }
        return std::pair{found, hit};
    };
    std::map<std::string, std::pair<std::size_t, std::size_t>> areas;
    for (const auto& [area, prefix] : kSettings.coverageAreas) {
        areas[area] = tally([&prefix](std::string_view name) { return name.starts_with(prefix); });
    }
    const auto [foundAll, hitAll]     = tally(inArea);
    const std::optional<double> total = percent(foundAll, hitAll);
    using ByFile                      = std::map<std::string, std::set<unsigned>, std::less<>>;
    ByFile           covered;
    ByFile           uncovered;
    ByFile           noData;
    const std::regex excluded{kSettings.coverageExcluded};
    const ByFile     changed = base.commit.empty() ? ByFile{} : changedLinesIn(run({"git", "diff", "-U0", "--no-color", base.commit + "...HEAD"}).out);
    for (const auto& [name, numbers] : changed) {
        if (!inArea(name) || std::regex_search("/" + name, excluded)) {
            continue;
        }
        const auto                     counts = lines.find(name);
        const std::vector<std::string> source = fs::exists(name) ? linesOf(readFile(name)) : std::vector<std::string>{};
        const bool                     isCpp  = name.ends_with(".cpp") || name.ends_with(".hpp") || name.ends_with(".h") || name.ends_with(".cc");
        for (const unsigned number : numbers) {
            if (counts != lines.end() && counts->second.contains(number)) {
                (counts->second.at(number) > 0U ? covered : uncovered)[name].insert(number);
            } else if (isCpp && number >= 1U && number <= source.size() && executableLooking(source[number - 1U])) {
                noData[name].insert(number);
            }
        }
    }
    const auto count = [](const ByFile& byFile, std::string_view prefix = "") {
        std::size_t counted = 0;
        for (const auto& [name, numbers] : byFile) {
            counted += name.starts_with(prefix) ? numbers.size() : 0U;
        }
        return counted;
    };
    const std::size_t           changedTotal   = count(covered) + count(uncovered);
    const std::size_t           changedCovered = count(covered);
    const std::optional<double> patch          = percent(changedTotal, changedCovered);
    const Row&                  previous       = base.row;
    const auto                  previousOf     = [&previous](std::string_view key) { return numberIn(valueOf(previous, key)); };
    const auto                  dotOf          = [](double share) { return share >= 90.0 ? kGreen : share >= kSettings.patchMinimum ? kYellow : kRed; };
    const std::string           project        = std::format("project {} %{}", shownPercent(total), previous.empty() ? std::string{} : std::format(" ({})", signedChange(total, previousOf("total"))));
    const std::string           headline       = changedTotal != 0U ? std::format("changed lines {} % ({} of {}) · {}", shownPercent(patch), changedCovered, changedTotal, project) : std::format("no changed executable lines · {}", project);
    const bool                  noGo           = patch && *patch < kSettings.patchMinimum;
    const std::string           reason         = noGo ? std::format("{} no test runs", plural(count(uncovered), "changed line")) : std::string{};
    const auto                  areaRow        = [&](const std::string& area, const std::string& prefix) {
        const std::size_t           changedHere = count(covered, prefix) + count(uncovered, prefix);
        const std::optional<double> share       = percent(changedHere, count(covered, prefix));
        const std::optional<double> areaShare   = percent(areas.at(area).first, areas.at(area).second);
        return std::format("| `{}` | {} | {} | {} | {} % |\n", prefix, changedHere != 0U ? std::to_string(changedHere) : "–", changedHere != 0U ? std::format("{} {} %", dotOf(*share), shownPercent(share)) : "–", signedChange(areaShare, previousOf(area)), shownPercent(areaShare));
    };
    const std::string tableHead = "| area | changed lines | covered | area Δ | area |\n|---|---:|---:|---:|---:|\n";
    std::string       touchedRows;
    std::string       allRows;
    for (const auto& [area, prefix] : kSettings.coverageAreas) {
        if (areas.at(area).first == 0U) {
            continue;
        }
        allRows += areaRow(area, prefix);
        touchedRows += count(covered, prefix) + count(uncovered, prefix) != 0U ? areaRow(area, prefix) : std::string{};
    }
    std::string body = touchedRows.empty() ? std::string{} : tableHead + touchedRows + "\n";
    body += trend("coverage", "line coverage of the project", base, "total", total, "%");
    if (!uncovered.empty()) {
        body += details(std::format("⚠️ {} changed lines not covered", count(uncovered)), listedLines(uncovered));
    }
    if (!noData.empty()) {
        body += details(std::format("❔ {} changed lines without coverage data (e.g. templates no test instantiates, code only a GPU or a browser runs; not counted)", count(noData)), listedLines(noData));
    }
    std::string allAreas = tableHead + allRows;
    allAreas += std::format("| **total** | {} | {} | {} | {} % |\n\n", changedTotal != 0U ? std::to_string(changedTotal) : "–", changedTotal != 0U ? std::format("{} {} %", dotOf(*patch), shownPercent(patch)) : "–", signedChange(total, previousOf("total")), shownPercent(total));
    allAreas += "| | main | this | Δ |\n|---|---:|---:|---:|\n";
    for (const std::string key : {"regions", "functions", "branches", "instantiations"}) {
        const std::optional<double> before = previousOf(key);
        const std::optional<double> now    = totals.at(key);
        allAreas += std::format("| {} | {} | {} | {} |\n", key == "instantiations" ? "instantiations *(informational)*" : key, before ? std::format("{:.1f} %", *before) : "n/a", now ? std::format("{:.1f} %", *now) : "n/a", signedChange(now, before));
    }
    body += details("All areas; regions, functions, branches, instantiations", allAreas);
    const std::vector<std::string> versionLines = linesOf(run({llvmCov, "--version"}).out);
    const auto                     versionLine  = std::ranges::find_if(versionLines, [](const std::string& line) { return line.contains("version"); });
    const std::string              version      = versionLine == versionLines.end() ? std::string{} : trimmed(*versionLine);
    body += details("Method", std::format("The native tests of the instrumented AdaptiveCpp build for its OpenMP host target ({}), {} profiles over {} binaries{}; tests, fixtures and dependencies excluded (`{}`). Code only a GPU or a browser runs counts as not covered. Changed lines: since {}, those the compiler made executable; no-go below {} %; the project's change is shown, not counted. {}", kSettings.coverageMethod, profiles.size(), objects.size(), rejected.empty() ? std::string{} : std::format(", {} more that llvm-cov rejects left out (`coverage-skipped.txt`)", rejected.size()), kSettings.coverageExcluded, base.commit.empty() ? "—" : base.commit.substr(0, 8), shortest(kSettings.patchMinimum), version));
    // `lines` is llvm-cov's share over every file not excluded, the figure coverage-data recorded; `total` the areas' only
    Figures figures{{"total", total}, {"patch", patch}};
    for (const auto& [area, prefix] : kSettings.coverageAreas) {
        figures.emplace_back(area, percent(areas.at(area).first, areas.at(area).second));
    }
    for (const std::string key : {"lines", "regions", "functions", "branches", "instantiations"}) {
        figures.emplace_back(key, totals.at(key));
    }
    figures.emplace_back("binaries", static_cast<double>(objects.size()));
    figures.emplace_back("skipped", static_cast<double>(rejected.size()));
    const std::string_view         dot        = changedTotal != 0U ? dotOf(*patch) : kGreen;
    const std::string              minimum    = shortest(kSettings.patchMinimum);
    const std::string              legend     = std::format("{} ≥ 90 % · {} ≥ {} % · {} < {} % of the changed lines", kGreen, kYellow, minimum, kRed, minimum);
    const std::string              pages      = links().pages;
    const std::vector<std::string> mainReport = pages.empty() ? std::vector<std::string>{} : std::vector<std::string>{std::format("[latest main report]({}/coverage/)", pages)};
    writeReport(arguments.out, "coverage", base, {.dot = dot, .headline = headline, .reason = reason, .body = body, .legend = legend, .figures = figures, .footerLinks = mainReport});
}

// instructions of each listed benchmark under callgrind (the whole run: its iterations dominate the set-up) and the
// shipped binaries' sizes; hosted runners grant no hardware counters, and instruction counts do not vary with the machine
void hotPaths(const Arguments& arguments) {
    const Base                     base   = baseline("hot-paths");
    const std::vector<std::string> listed = wordsOf(readFile(kRepository / "devtools" / "ci" / "hot-path-benchmarks.txt"));
    std::vector<fs::path>          benchmarks;
    for (const auto& entry : fs::recursive_directory_iterator(arguments.build)) {
        if (entry.is_regular_file() && std::ranges::contains(listed, entry.path().filename().string()) && (entry.status().permissions() & fs::perms::owner_exec) != fs::perms::none && isOwnBuildFile(entry.path())) {
            benchmarks.push_back(entry.path());
        }
    }
    std::ranges::sort(benchmarks, {}, [](const fs::path& path) { return path.filename(); });
    for (const std::string& name : listed) {
        if (std::ranges::none_of(benchmarks, [&name](const fs::path& path) { return path.filename() == name; })) {
            std::println(stderr, "{} is listed but not built in {}", name, arguments.build.string());
            std::exit(2);
        }
    }
    Figures figures;
    for (const fs::path& benchmark : benchmarks) {
        const fs::path out = arguments.out / (benchmark.filename().string() + ".callgrind");
        // a benchmark that stops early still leaves a profile, of its start-up only: such a figure must not enter the history
        if (const Ran ran = run({"valgrind", "--tool=callgrind", "--callgrind-out-file=" + out.string(), benchmark.string()}); ran.status != 0) {
            const std::vector<std::string> errors = linesOf(ran.err);
            const std::string              tail   = joined(errors | std::views::drop(errors.size() > 20U ? errors.size() - 20U : 0U), "\n");
            std::println(stderr, "{} under callgrind exited {}:\n{}", benchmark.string(), ran.status, tail);
            std::exit(2);
        }
        const std::vector<std::string> profile = linesOf(readFile(out));
        const auto                     total   = std::ranges::find_if(profile, [](const std::string& line) { return line.starts_with("summary: ") || line.starts_with("totals: "); });
        if (total == profile.end()) {
            std::println(stderr, "callgrind counted no instructions of {} (is valgrind installed?)", benchmark.string());
            std::exit(2);
        }
        figures.emplace_back(benchmark.filename().string(), numberIn(total->substr(total->find(' ') + 1)));
    }
    for (const std::string& binary : kSettings.hotPathBinaries) {
        if (fs::exists(arguments.build / binary)) {
            figures.emplace_back(fs::path{binary}.filename().string() + " bytes", static_cast<double>(fs::file_size(arguments.build / binary)));
        }
    }
    const double                                 limit = kSettings.hotPathLimit;
    const double                                 noise = kSettings.hotPathNoise;
    std::map<std::string, std::optional<double>> changes;
    for (const auto& [name, value] : figures) {
        const std::optional<double> recorded = numberIn(valueOf(base.row, name));
        changes[name]                        = recorded && *recorded != 0.0 && value ? std::optional<double>{100.0 * (*value - *recorded) / *recorded} : std::nullopt;
    }
    const auto dotOf = [&](std::optional<double> change) { return !change || std::abs(*change) < noise ? std::string_view{} : *change > limit ? kRed : *change > 0.0 ? kYellow : kGreen; };
    const auto row   = [&](const std::string& name, double value) {
        const std::optional<double> change = changes.at(name);
        return std::format("| {} | {} | {} | {} |\n", dotOf(change), name, !change ? std::string{"n/a"} : std::abs(*change) < noise ? std::string{"±0"} : std::format("{:+.2f} %", *change), withThousands(std::llround(value)));
    };
    const std::string        tableHead = "| | figure | Δ | now |\n|---|---|---:|---:|\n";
    std::vector<std::string> moved;
    std::size_t              above  = 0;
    std::size_t              rising = 0;
    for (const auto& [name, value] : figures) {
        const std::string_view dot = dotOf(changes.at(name));
        if (!dot.empty()) {
            moved.push_back(name);
        }
        above += dot == kRed ? 1U : 0U;
        rising += dot == kYellow ? 1U : 0U;
    }
    std::ranges::stable_sort(moved, std::greater{}, [&](const std::string& name) { return std::abs(*changes.at(name)); });
    const auto  valueOfFigure = [&figures](const std::string& name) { return std::ranges::find(figures, name, &Figures::value_type::first)->second.value_or(0.0); };
    std::string body;
    if (!moved.empty()) {
        body += tableHead;
        for (const std::string& name : moved) {
            body += row(name, valueOfFigure(name));
        }
        body += "\n";
    }
    const auto        isBenchmark = [](const std::string& name) { return name.starts_with("bm_"); };
    const auto        movedMost   = std::ranges::find_if(moved, isBenchmark);
    const auto        firstListed = std::ranges::find_if(figures, isBenchmark, &Figures::value_type::first);
    const std::string firstName   = firstListed != figures.end() ? firstListed->first : std::string{};
    const std::string charted     = movedMost != moved.end() ? *movedMost : firstName;
    if (!charted.empty()) {
        body += trend("hot-paths", std::format("{} instructions", charted), base, charted, valueOfFigure(charted), "instructions");
    }
    std::string all = tableHead;
    for (const auto& [name, value] : figures) {
        all += row(name, value.value_or(0.0));
    }
    body += details("All figures", all);
    body += details("Method", std::format("Instructions: each benchmark listed in `devtools/ci/hot-path-benchmarks.txt`, from the GCC 15 Release build with `-march=x86-64-v3`, run once under `valgrind --tool=callgrind`, the whole process counted (its iterations dominate the set-up); the listed ones repeat within 0.015 % (2026-10-16), so a change below {} % reads ±0; wall time is not compared. "
                                          "Sizes: the release binaries as built. A rise above {} % is a no-go unless the commit message names it (CLAUDE.md, measured quality).",
                                  shortest(noise), shortest(limit)));
    std::string      headline;
    std::string_view dot;
    if (base.row.empty()) {
        headline = std::format("{} measured", plural(figures.size(), "figure"));
    } else {
        const std::string verdict = above != 0U ? std::format("{} up by more than {} %", above, shortest(limit)) : rising != 0U ? std::format("{} up by at most {} %", rising, shortest(limit)) : std::string{"no figure up"};
        const std::string largest = moved.empty() ? std::string{} : std::format(" · `{}` {:+.1f} %", moved.front(), *changes.at(moved.front()));
        headline                  = verdict + largest;
        dot                       = above != 0U ? kRed : rising != 0U ? kYellow : kGreen;
    }
    const std::string reason = above != 0U ? "name it in the commit message" : "";
    const std::string legend = std::format("{} down · {} up by at most {} % · {} up by more than {} % · ±0 within {} %", kGreen, kYellow, shortest(limit), kRed, shortest(limit), shortest(noise));
    writeReport(arguments.out, "hot-paths", base, {.dot = dot, .headline = headline, .reason = reason, .body = body, .legend = legend, .figures = figures});
}

void wasmSize(const Arguments& arguments) {
    Figures     sizes;
    std::string body;
    double      sum = 0.0;
    for (const std::string& module : kSettings.wasmModules) {
        if (fs::exists(arguments.build / module)) {
            const auto bytes = fs::file_size(arguments.build / module);
            sizes.emplace_back(fs::path{module}.filename().string() + " bytes", static_cast<double>(bytes));
            body += std::format("- `{} bytes`: {}\n", fs::path{module}.filename().string(), withThousands(static_cast<long long>(bytes)));
            sum += static_cast<double>(bytes);
        }
    }
    writeReport(arguments.out, "wasm-size", Base{}, {.dot = {}, .headline = std::format("{} bytes", withThousands(std::llround(sum))), .reason = {}, .body = body, .legend = {}, .figures = sizes});
}

// replaces the ci-metrics branch by one parentless commit: each bot's history with this commit's row (a rerun's in its
// place) in commit order, and its history report for the last rows; a lease on the fetched head makes a writer that lost
// the race to another main run start over
void record(const Arguments& arguments) {
    struct Recorded {
        std::string              bot;
        Row                      row;
        std::vector<std::string> figureNames;
        fs::path                 report;
    };
    const std::string branch = kSettings.historyRef.substr(kSettings.historyRef.find('/') + 1);
    const std::string commit = environmentOr("GITHUB_SHA", trimmed(checked({"git", "rev-parse", "HEAD"}).out));
    // the merged pull request as GitHub knows it (needs GH_TOKEN): main is rebase-merged, so subjects name none, or a wrong one
    const std::string repository = environmentOr("GITHUB_REPOSITORY", "");
    const auto        mergedBy   = [&] {
        const Ran pulls = run({"gh", "api", std::format("repos/{}/commits/{}/pulls", repository, commit), "--jq", ".[0].number // empty"});
        return pulls.status == 0 ? trimmed(pulls.out) : std::string{};
    };
    const std::string     date        = trimmed(checked({"git", "log", "-1", "--format=%cI", commit}).out);
    const std::string     pullRequest = repository.empty() ? std::string{} : mergedBy();
    const Row             common{{"commit", commit}, {"date", date}, {"pr", pullRequest}};
    std::vector<Recorded> recorded;
    for (const fs::path& directory : arguments.reportDirectories) {
        const Table figures = csvTable(readFile(directory / "figures.csv"));
        if (figures.rows.empty()) {
            std::println(stderr, "no figures in {}", directory.string());
            std::exit(2);
        }
        const Row&     measured = figures.rows.front();
        const fs::path report   = directory / kHistoryReport;
        Recorded       entry{.bot = measured.at("bot"), .row = common, .figureNames = {figures.columns.begin() + 2, figures.columns.end()}, .report = fs::exists(report) ? report : fs::path{}};
        entry.row["tool"] = measured.at("tool");
        for (const std::string& column : entry.figureNames) {
            entry.row[column] = measured.at(column);
        }
        recorded.push_back(std::move(entry));
    }
    if (recorded.empty()) {
        std::println(stderr, "no report directory to record");
        std::exit(2);
    }
    const std::vector<std::string> environment{"GIT_AUTHOR_NAME=" + kSettings.author, "GIT_AUTHOR_EMAIL=41898282+github-actions[bot]@users.noreply.github.com", "GIT_COMMITTER_NAME=" + kSettings.author, "GIT_COMMITTER_EMAIL=41898282+github-actions[bot]@users.noreply.github.com"};
    const fs::path                 index   = arguments.out / "record.index";
    std::vector<std::string>       indexed = environment;
    indexed.push_back("GIT_INDEX_FILE=" + index.string());
    const auto stage = [&indexed](const std::string& path, std::string_view content) {
        const std::string blob = trimmed(checked({"git", "hash-object", "-w", "--stdin"}, content).out);
        checked({"git", "update-index", "--add", "--cacheinfo", std::format("100644,{},{}", blob, path)}, {}, indexed);
    };
    const std::string bots = joined(recorded | std::views::transform(&Recorded::bot), ", ");
    std::mt19937      jitter{std::random_device{}()};
    for (int attempt = 0; attempt < 8; ++attempt) {
        const bool        fetched = run({"git", "fetch", "--no-tags", "origin", "refs/heads/" + branch}).status == 0;
        const std::string parent  = fetched ? trimmed(checked({"git", "rev-parse", "FETCH_HEAD"}).out) : std::string{};
        fs::remove(index);
        if (!parent.empty()) {
            checked({"git", "read-tree", parent}, {}, indexed);
        }
        for (const Recorded& entry : recorded) {
            Table      table   = parent.empty() ? Table{} : csvTable(run({"git", "show", parent + ":" + entry.bot + "/history.csv"}).out);
            const auto isRerun = [&](const Row& kept) { return valueOf(kept, "commit") == commit && valueOf(kept, "tool") == entry.row.at("tool"); };
            const auto rerun   = std::ranges::find_if(table.rows, isRerun);
            if (rerun == table.rows.end()) {
                table.rows.push_back(entry.row);
            } else {
                *rerun = entry.row;
            }
            // commit order, also when an older commit is recorded late
            std::ranges::stable_sort(table.rows, {}, [](const Row& kept) { return epochSecondsOf(valueOf(kept, "date")).value_or(0LL); });
            std::vector<std::string> columns{"commit", "date", "pr", "tool"};
            for (const std::vector<std::string>& names : {table.columns, entry.figureNames}) {
                for (const std::string& column : names) {
                    if (!std::ranges::contains(columns, column)) {
                        columns.push_back(column);
                    }
                }
            }
            table.columns = columns;
            stage(entry.bot + "/history.csv", csvText(table));
            if (entry.report.empty()) {
                continue;
            }
            stage(std::format("{}/reports/{}.txt", entry.bot, commit), readFile(entry.report));
            std::set<std::string> kept;
            for (const Row& keptRow : table.rows | std::views::drop(table.rows.size() > kSettings.keptReports ? table.rows.size() - kSettings.keptReports : 0U)) {
                kept.insert(std::format("{}.txt", valueOf(keptRow, "commit")));
            }
            const std::string              reports = parent + ":" + entry.bot + "/reports";
            const std::vector<std::string> stored  = parent.empty() ? std::vector<std::string>{} : wordsOf(run({"git", "ls-tree", "--name-only", reports}).out);
            for (const std::string& name : stored | std::views::filter([&kept](const std::string& name) { return !kept.contains(name); })) {
                checked({"git", "update-index", "--force-remove", std::format("{}/reports/{}", entry.bot, name)}, {}, indexed);
            }
        }
        const std::string tree    = trimmed(checked({"git", "write-tree"}, {}, indexed).out);
        const std::string message = std::format("metrics of {}: {}", commit.substr(0, 7), bots);
        const std::string created = trimmed(checked({"git", "commit-tree", tree, "-m", message}, {}, environment).out);
        if (run({"git", "push", std::format("--force-with-lease=refs/heads/{}:{}", branch, parent), "origin", created + ":refs/heads/" + branch}).status == 0) {
            std::println("recorded {} for {} as {}", bots, commit.substr(0, 7), created.substr(0, 7));
            return;
        }
        const auto delay = std::chrono::milliseconds{2000 + attempt * 3000 + static_cast<int>(jitter() % 3000U)};
        std::println("{} moved on; trying again in {} s", branch, delay.count() / 1000);
        std::this_thread::sleep_for(delay);
    }
    std::println(stderr, "could not record {} for {}: {} kept moving", bots, commit.substr(0, 7), branch);
    std::exit(1);
}

constexpr std::string_view kUsage = R"(usage: qualitybot <formatting|naming|code-style|coverage|hot-paths|wasm-size> --out <dir> [--build <dir>] [--profiles <dir>]
       qualitybot record --out <dir> <report dir>...
  --out       where comment.md, status.json and figures.csv go (record: its scratch files)
  --build     the build directory (naming, code-style: Clang; coverage, hot-paths, wasm-size: their flavour's)
  --profiles  the directory of .profraw files the tests wrote (coverage)
  <report dir> a bot's --out, whose figures.csv and history-report.txt record adds to the ci-metrics branch)";

} // namespace

int main(int argc, char** argv) {
    const std::vector<std::string_view> words(argv + 1, argv + argc);
    Arguments                           arguments;
    for (std::size_t i = 0; i < words.size(); ++i) {
        const auto value = [&] { return i + 1 < words.size() ? fs::path{words[++i]} : fs::path{}; };
        if (words[i] == "--out") {
            arguments.out = value();
        } else if (words[i] == "--build") {
            arguments.build = value();
        } else if (words[i] == "--profiles") {
            arguments.profiles = value();
        } else if (arguments.bot.empty()) {
            arguments.bot = std::string{words[i]};
        } else {
            arguments.reportDirectories.emplace_back(words[i]);
        }
    }
    const std::map<std::string_view, void (*)(const Arguments&)> kRun = {{"formatting", formatting}, {"naming", naming}, {"code-style", codeStyle}, {"coverage", coverage}, {"hot-paths", hotPaths}, {"wasm-size", wasmSize}, {"record", record}};
    const auto                                                   bot  = kRun.find(arguments.bot);
    if (bot == kRun.end() || arguments.out.empty()) {
        std::println(stderr, "{}", kUsage);
        return 2;
    }
    fs::create_directories(arguments.out);
    bot->second(arguments);
    return 0;
}
