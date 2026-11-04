// code-metrics: code-shape figures of this tree, read through libclang; kMeasurementMethod below says how each is
// measured and is what --help prints

#include "Ccd.hpp"
#include "Figures.hpp"
#include "Libclang.hpp"

#include <clang-c/Index.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <format>
#include <fstream>
#include <iterator>
#include <map>
#include <mutex>
#include <print>
#include <ranges>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <tuple>
#include <vector>

namespace {

namespace fs = std::filesystem;
using devtools::fileOf;
using devtools::firstChildOf;
using devtools::taken;

constexpr std::string_view kMeasurementMethod = R"method(code-metrics: how hard is this code to read and to change, and did my change make it easier or harder?

quick start, from the repository root, with a Clang build (cmake-build-Clang22-Debug below):
  devtools/code-metrics.sh cmake-build-Clang22-Debug --cognitive --list > before.txt          # on the tree before your change
  # make the change, rebuild
  devtools/code-metrics.sh cmake-build-Clang22-Debug --cognitive --compare before.txt         # what moved, where, and the usual remedy
  devtools/code-metrics.sh cmake-build-Clang22-Debug --cognitive --compare before.txt --table # the same figures as a table, for the commit message

usage: code-metrics <clang-build-dir> --root <dir>... [--exclude <dir>]... [--core <dir>] [--jobs <n>] [--list]
                    [--cognitive-scores <file>] [--summary | --table | --compare <earlier report> [--table] [--list]]
       code-metrics --figures | --help
  --root <dir>               measure the files under <dir>, relative to the repository (the working directory); repeatable;
                             a file with a test/ or benchmarks/ directory in its path is a test, every other one a source
  --exclude <dir>            leave out a directory under a root (repeatable)
  --core <dir>               the part whose functions with ambient reach are counted apart
  --list                     every figure with the places behind it, one per line beneath it: "<value> | <key> | <where>";
                             the key names a file, function or #include edge without a line number, so it is found again later;
                             keep this report: --compare reads it back
  --summary                  the figures only, each with its meaning
  --table                    the figures as a Markdown table, for a pull request or a commit message
  --compare <report>         each figure before (from the earlier report) and now, with its change and whether it moved the better way;
                             from an earlier --list report also, under each figure that moved, the places that moved it (new, gone, rose, fell;
                             the largest 5, all with --list) and, when it moved the worse way, the usual remedy ("do:" when mechanical,
                             "usual way:" when it needs judgement); with --table the figures only. Never fails: figures are evidence, not a gate
  --figures                  the figures, their unit and their better direction
devtools/code-metrics.sh runs this and adds cognitive complexity (--cognitive), hot paths and seeded faults.

How to read the figures: compare two states of the tree; no figure is read alone, and none says by itself whether code is easy
to understand. Each figure counts one kind of cost a reader pays to know what a change does: how far it reaches (1), how much
must be held in mind while reading (2), and how much there is to read (3). A figure that rose is a question for the review, not a
verdict: the card says when a rise is the design ("ignore when") and what usually lowers it ("do").
The tool reads compile_commands.json of a Clang build and parses every translation unit under the roots once with libclang;
calls are resolved to their definitions, so a jump is a real jump, not a name match.
Each card: asks (the question the figure answers) · score (how it is counted) · rises when · ignore when · do · example ·
where (what --list and --compare show) · method (origin, choices, how it is fooled, references).

(1) composition: how far a change can reach

CCD, NCCD, include edges   how tangled the #includes are                                         lower is better; NCCD 1 is tidy
  asks        when I open one file, how many other files must I understand before I can trust it?
  score       per component (a file stem: Markdown.hpp and Markdown.cpp are one), every component its #includes reach, itself
              counted; summed over all components (CCD); divided by the CCD of a balanced tree of as many components (NCCD)
  rises when  a header gains an #include: every file that includes that header now reaches the new one and all it reaches;
              a file is split into two that include each other's parts
  ignore when the edge is the design: a system's entry point included by main.cpp, or the core's public API
  do          when a header uses a type only by reference or pointer, it declares the type (struct Theme;) and its .cpp includes it
  example     three files in a chain, each including the next:
                // a.hpp
                #include "b.hpp"        // a reaches a, b, c  -> 3
                // b.hpp
                #include "c.hpp"        // b reaches b, c     -> 2
                // c.hpp                // c reaches c        -> 1
                                        // CCD = 3 + 2 + 1 = 6; a balanced tree of 3 has CCD 5, so NCCD = 6 / 5 = 1.2
              the same three files including only the standard library: CCD = 1 + 1 + 1 = 3, NCCD = 3 / 5 = 0.6.
              Below 1 is flatter than a balanced tree, above 1 more tangled. The usual way to lower it:
                // Plot.hpp, before                             // Plot.hpp, after
                #include "Theme.hpp"                            struct Theme;                   // a reference needs only a declaration
                void drawPlot(const Theme& theme);              void drawPlot(const Theme& theme);
                                                                // Plot.cpp: #include "Theme.hpp"
              every component that includes Plot.hpp now reaches one component fewer.
  where       --list prints each edge "Plot -> Theme" with how much CCD removing it alone saves; --compare names the edges that are
              new or gone. "widest reach: ViewerExport 57" names the components that reach the most others.
  method      J. Lakos, Large-Scale C++ Software Design, 1996, ch. 4 [1]; #include edges only, not calls; external libraries left
              out; under the source roots. Fooled by a header that forwards everything (one edge hides many) and by merging files.

cycle       components that reach themselves through #include                                    lower is better, 0 wanted
  asks        are there two files that cannot be understood, tested or deleted one without the other?
  score       1 per component that reaches itself through #include
  rises when  two headers start including each other, directly or through others
  ignore when never: a cycle is always worth breaking
  do          one of the two headers declares the other's type instead of including it
  example     // Slides.hpp
              #include "Viewer.hpp"     // Slides reaches Viewer
              // Viewer.hpp
              #include "Slides.hpp"     // and Viewer reaches Slides: components in a cycle = 2
              Slides.hpp only passes a Viewer by reference, so a declaration is enough:
              // Slides.hpp
              struct Viewer;            // no #include: Slides no longer reaches Viewer, components in a cycle = 0
  where       --list names each component in a cycle; --compare names the ones that joined or left
  method      [1], ch. 4 (levelisation); a forward declaration is not an edge.

library reach   translation units under the source roots that include each external library                  read, not optimised
  asks        how many of the viewer's files break, or rebuild, when GR4, ImGui or SDL changes?
  score       per library, the translation units under the source roots that include it through any depth of #include; a library is named
              by its _deps/<name>-src directory, the standard library "std", other headers under /usr "system"
  rises when  a widely included header includes a library for one small need
  do          include the library in the .cpp that uses it, not in a header others include
  example     the report line "gnuradio4 41, imgui 32, sdl3 18 (62 units)" says GR4 reaches 41 of the 62 translation units.
              // Markdown.hpp
              #include <imgui.h>        // added for one colour type
              Now every unit that includes Markdown.hpp reaches ImGui, although a Markdown parser does not draw: the imgui
              count rises by the number of those units, and each of them rebuilds when ImGui changes.
  where       one line in every report; not compared

calls across, lone callees   how often reading means opening another file                       lower is better
  asks        to follow what a function does, how often must I open another file, and which of those jumps buy nothing?
  score       calls across: the share of calls to the project's own functions that land in another component;
              lone callees: functions called from exactly one place, in another component, and by no test
  rises when  a piece of a function moves into another file although nothing else calls it
  ignore when the callee is a system's entry point called from the frame loop (main -> drawCurrentSection), or the core's API:
              those jumps are the design (CLAUDE.md §2)
  do          a lone callee is inlined at its one caller, or gets a qa_ test, which makes its name a contract one can trust
              without reading the body; a function with one caller in another file moves next to that caller
  example     // DocumentView.cpp
              void drawDocument(const Document& document) {
                  drawPlot(document.plot);  // defined in PlotView.cpp: a call across components
                  layoutOf(document);       // defined in this file: not across
              }
              // PlotView.cpp
              void drawPlot(const Plot& plot) { ... }   // called only from drawDocument, and by no test: a lone callee
              Here 1 of 2 calls leaves its component (50 %). slice() in Slice.hpp, with many callers and qa_Slice, is a call
              across but never a lone callee.
  where       --list prints the component pairs "DocumentView -> PlotView" with their call counts, and each lone callee with
              its one caller; --compare names the pairs that rose and the lone callees that are new or gone
  method      CLAUDE.md §2, "the behaviour behind a call site one jump away"; C. Muratori, "Semantic Compression" [3];
              "test" means called from test/, not asserted on; calls through std::function, function pointers and virtual
              dispatch are not resolved.

ambient reach   what a function depends on that its signature does not show                    more functions without is better
  asks        can a test call this function with values and know the answer, or does it also read a clock, a global or the screen?
  score       the share of source functions that reach none of: a non-const variable of static storage (namespace scope, static
              member, function-local static, thread_local), the drawing context (ImGui, ImPlot, SDL, GL, emscripten), the clock
              (std::chrono::*::now), the environment (getenv), input and output (files, std::filesystem beyond path arithmetic,
              cpr, fopen); a function reaches whatever its own callees reach; core functions that reach any are counted apart
  rises when  a function starts reading the time, a setting or a global itself instead of being given it
  ignore when drawing code reads the drawing context: that is its job
  do          read the clock, environment or global once where the frame starts and pass the value down as a parameter;
              the core reads nothing ambient at all
  example     double frameStart() { return ImGui::GetTime(); }        // drawing context: a test cannot choose the time
              double age(double now) { return now - frameStart(); }   // drawing context too, through its callee
              int nextId() { static int next = 0; return ++next; }    // global: the result depends on all earlier calls
              The same without ambient reach takes what it reads as a parameter:
              double age(double now, double frameStartSeconds) { return now - frameStartSeconds; }   // none
              and the frame loop reads ImGui::GetTime() once and passes it down, so a test can call age(5.0, 3.0) and expect 2.0.
  where       --list prints each function that reaches something, with what it reaches ("directly: getenv" when it calls it
              itself); --compare names the functions that started or stopped reaching something
  method      CLAUDE.md §2, "ambient input is read once where the frame starts and passed down as data"; printing to stdout or
              stderr and gr::log are write-only and not counted; indirect calls are not resolved, so reach is a lower bound.

unused functions   code nothing reaches                                                          lower is better
  asks        which functions are defined but named by no code here, so a reader studies what never runs?
  score       1 per function defined under the source roots that no own code names, tests included: not called, not passed, not taken
              by address; constructors, destructors, operators, virtual functions and main are reached without a name
  rises when  the last caller of a function goes and the function stays
  ignore when the function is a library API for callers outside this repository
  do          delete it (CLAUDE.md §4: dead API goes)
  where       --list prints each function with its line and length
  method      this repository; a function named only in code the build's preprocessor leaves out (#ifdef __EMSCRIPTEN__ in a
              native build) is listed.

(2) cognitive complexity: how much a reader must hold

cognitive complexity   summed over the sources (code-metrics.sh --cognitive)                            lower is better
  asks        how many branches and loops must a reader keep in mind to follow the code, summed over all functions?
  score       +1 for each break in the straight reading: if, else if, else, ?:, switch, loop, catch, and each run of one
              operator, && or ||, in a condition; +1 more for each level of nesting an if, ?:, switch, loop or catch sits in
  rises when  a branch or loop is added; a condition moves one level deeper; a decision is nested inside a loop
  ignore when a parser or a state machine has that many cases by nature: name the number in the commit message
  do          return early instead of else; join nested ifs with &&; a chain of ifs on one enum becomes a switch
  example     void drawItems(bool ready, std::span<const Item> items) {
                  if (ready) {                              // +1  an if at depth 0
                      for (const Item& item : items) {      // +2  a loop: 1, plus 1 for sitting inside the if
                          if (item.visible) {               // +3  an if: 1, plus 2 for sitting inside the if and the loop
                              draw(item);
                          }
                      }
                  } else {                                  // +1  an else: no nesting penalty, it is at the if's own depth
                      wait();
                  }
              }                                             // cognitive complexity = 1 + 2 + 3 + 1 = 7
              With an early return the nesting goes, and so does the else:
              void drawItems(bool ready, std::span<const Item> items) {
                  if (!ready) { wait(); return; }           // +1
                  for (const Item& item : items) {          // +1  now at depth 0
                      if (item.visible) { draw(item); }     // +2  nested once
                  }
              }                                             // 1 + 1 + 2 = 4
              Conditions count runs of the same operator, not each operator:
              if (a && b && c)          // +1 for the if, +1 for one run of &&                       = 2
              if (a && b || c && d)     // +1 for the if, +3 for the runs &&, then ||, then &&       = 4
  where       --list prints each function's score; --compare names the functions that rose or fell, by how much
  method      G. A. Campbell, "Cognitive Complexity", SonarSource, 2017 [2], as clang-tidy's readability-function-cognitive-
              complexity scores it; summed, never the maximum alone, because the maximum rewards splitting a function without
              simplifying it. Fooled by moving a nested loop into its own function: the sum falls, the reading does not (next card).

split-invariant complexity   the same sum, single-use pieces folded back                      lower is better
  asks        did splitting a function into pieces make it simpler, or only spread the same reading over more places?
  score       cognitive complexity with every callee that has one call site under the source roots and no test folded into its caller:
              its score, plus the call's nesting depth for each of its increments that carry a nesting penalty
  rises when  complexity is added, whether straight or in a single-use piece
  falls when  complexity is removed; never by splitting alone
  do          a single-use piece split off gets a qa_ test or a second caller, or is folded back into its caller
  example     the first function of the previous card, with its loop moved into a function called once and by no test:
              void drawItems(bool ready, std::span<const Item> items) {
                  if (ready) {                              // +1
                      drawVisible(items);                   // the call sits at depth 1
                  } else {                                  // +1
                      wait();
                  }
              }
              void drawVisible(std::span<const Item> items) {
                  for (const Item& item : items) {          // +1, an increment with a nesting penalty
                      if (item.visible) { draw(item); }     // +2, an increment with a nesting penalty
                  }
              }
              Plain cognitive complexity: 2 + 3 = 5, lower than the 7 of the straight version, yet nothing got simpler.
              Folded back: 1 + 1 + (1 + 2) + 2 = 7, because each of drawVisible's two penalised increments sits one level deeper
              at its call. If drawVisible had a test or a second caller, it would keep its own 3 and the split would count.
  where       --list prints each function's folded score with its plain one ("before folding"); --compare names those that moved.
              A function whose plain score fell while its folded one did not was split, not simplified.
  method      this repository, after [3]; a tested or shared function keeps its own score, because its name can be trusted
              without reading it; devtools/test/code-metrics checks the folding against clang-tidy's score of hand-inlined twins.

live mutable locals   variables whose value depends on which line one is reading                 lower is better
  asks        how many variables must a reader track through a function, because they hold different values at different lines?
  score       1 per local written in two or more top-level statements of one function (a nested lambda counts on its own);
              parameters count too; an initialiser is not a write
  rises when  a variable is reused for a second value, or built up across a function
  ignore when it is a scanner's cursor or a state machine's state (CLAUDE.md §2)
  do          one const local per value, or a block that owns the writes
  example     int total = 0;                // declared and initialised: not yet a write
              total = basePrice();          // written in this statement
              log(total);                   // a reader must know which value total holds here
              total += tax(total);          // and written again: total is a live mutable local
              The same as named constants, each holding one value for its whole life:
              const int price = basePrice();
              log(price);
              const int total = price + tax(price);   // no live mutable local
              A loop is one statement, so the running sum below is not counted, although it is mutable:
              for (const Item& item : items) { total += item.price; }
  where       --list prints each such local with its function and the number of statements writing it
  method      CLAUDE.md §2, "const by default, block-scoped state"; state a loop carries is not seen.

write-only fields   state nobody looks at                                                   lower is better
  asks        which fields are set, or not even set, yet never read, so they cost a reader for nothing?
  score       1 per field declared under the source roots that no own code reads, tests included; a read is any use but the left side of `=`,
              a designated initialiser or a constructor's member initialiser (so += and ++ read); a record with a defaulted
              == or <=> reads every field, and its fields are not counted
  rises when  state is kept "for later", or its last reader goes
  ignore when the field is read by reflection (GR_MAKE_REFLECTABLE) or by a library through its address
  do          delete it, and its writes
  example     struct Pass { float scale; float trailingGap = 0.0f; };   // trailingGap written by every block, read by none
  where       --list prints Record::field and its line
  method      this repository; a field passed by reference counts as read, so one only written through a reference is missed;
              code the build's preprocessor leaves out (#ifdef __EMSCRIPTEN__ in a native build) is not read, so a field read only there is listed.

nested ternaries   a choice inside a choice in one expression                                   lower is better
  asks        where must a reader hold two conditions at once to read one value?
  score       1 per ?: that holds another ?: in the same expression (a lambda's body inside it is its own)
  rises when  a value grows a second case where a named step would say it
  do          name the inner choice as a const local, or use a switch or a small function (CLAUDE.md §2: one expression per idea)
  example     const float size = heading ? (level == 1 ? title : subtitle) : body;   // 1
              The same, named:   const float headingSize = level == 1 ? title : subtitle;
                                 const float size        = heading ? headingSize : body;   // 0
  where       --list prints each function with how many such ?: it holds
  method      CLAUDE.md §2, "one expression per idea: a nested ternary … SHOULD become named const locals".

flag arguments   a true or false whose meaning only the signature knows                            lower is better
  asks        which calls pass a bare true or false, so a reader must open the function to know what it switches?
  score       1 per parameter of an own function that some call fills with a literal true or false
  rises when  a function grows a mode instead of a second function, or a bool where an enum would name the choice
  ignore when the literal is a value, not a mode (setVisible(true)), or the call names it (/*paint=*/true is not seen)
  do          an enum class naming the modes, two functions, or a const bool local named for what it says at the call
  example     layoutPass(theme, document, scale, false, where, body);   // false: measuring, not painting
  where       --list prints the function, the parameter and how many calls pass true and false
  method      this repository; the argument's tokens as written.

same argument everywhere   a parameter that never varies                                   lower is better
  asks        which parameters does every caller fill with the same constant, so the function has one value it pretends to take?
  score       1 per parameter of an own function called from two places or more where every call passes the same literal,
              kConstant or qualified enumerator (Selection::all); a variable passed by name is not a constant
  rises when  a parameter is added "for later", or the last caller that varied it goes
  ignore when the function is a library API whose callers live elsewhere, or the parameter is a test's seam
  do          drop the parameter and use the value inside, as placementFor lost its always-1 scale
  example     float placementFor(const Document& document, float scale);   // scale: 1.0f at both calls
              placementFor(slide, 1.0f);
              placementFor(box, 1.0f);
  where       --list prints the function, the parameter, the value and the number of calls
  method      this repository; the argument's tokens as written, so 1.0f and 1.f differ; default arguments count as written.

unscoped pairs   what a function opens and must close by hand                                lower is better
  asks        how many things that must be closed are closed by a line a later edit can skip?
  score       1 per opening call matched by its closing call in the same function (Push…Pop, Begin…End, glGen…glDelete,
              Lock…Unlock, acquire…release, open…close, malloc…free, matched by the rest of the name), +1 per return, throw,
              break or continue between them that leaves the pair, +1 when the two sit in different branches
  rises when  a push and its pop are written out, or an early return is added between them
  ignore when the closing call is a scope object's destructor: then the pair is not in one function and is not counted
  do          a scope object closes it (a c_resource alias beside ScopedWindow, ScopedFont)
  example     list.PushClipRect(low, high, true);
              if (done) { list.PopClipRect(); return; }   // 1 for the pair, +1 for the return between them
              list.PopClipRect();
              The same as a scope object: no pair is written out, so nothing counts:
              const ScopedClipRect inStage{&list, low, high};
              if (done) { return; }
  where       --list prints each pair with its function, its exits and whether it is split across branches
  method      this repository; a name table, not a flow analysis: a pop with a count (PopStyleVar(2)) closes one, and a
              pair split across two functions is not seen.

(3) reading volume: how much there is to read

tokens + comment words, SLOC   everything there is to read                                      lower is better; SLOC reported only
  asks        how much text must a reader's eyes pass over, whatever its layout?
  score       every token and every comment word (a run of two or more letters) of the files under the source roots;
              SLOC counts their lines that hold code, and is reported, not optimised
  rises when  code or comments are added; a second implementation of something the codebase already has
  ignore when a feature adds what it needs: report the number in the commit message
  do          delete what restates the code; reuse what exists instead of writing it again
  example     // the nice-number rule                             // 4 comment words (the, nice, number, rule), 0 SLOC
              const double step = niceStep(range / kTickCount);   // 11 tokens, 1 SLOC
              The 11 tokens are const, double, step, =, niceStep, (, range, /, kTickCount, ), and ;. Spread over three lines,
              the statement is still 11 tokens but 3 SLOC, which is why SLOC is reported, not optimised. Renaming kTickCount to
              kNumberOfTicksOnTheAxis costs nothing here; a comment that says it again costs its words.
  where       --list prints each file's count; --compare names the files that grew or shrank most
  method      tokens, not lines or characters, so formatting and long names cost nothing; each file counted once, from the
              first unit that includes it. Fooled by a dense one-liner (cognitive complexity and long lines catch it).

repeated expressions   one value written out again and again                                   lower is better
  asks        which values are computed in several places from the same words, so that a change must find them all?
  score       per file, each expression of 8 tokens or more written 3 times or more, counted beyond its first writing;
              a part repeated only as often as a larger expression holding it counts as that expression
  rises when  a value is copied instead of named, as `image.empty() ? frame : region` was 13 times; a chain of names
              (viewer.navigator.cursor.step) is a reference, not a value, and does not count
  ignore when the repetition is the point: a table of similar calls, a test's arrange steps
  do          give the value a name: a const local, or a member function when the expression reads one type's fields
  example     const float left  = box.x + box.width * 0.5f - half;   // box.x+box.width*0.5f, written three times
              const float right = box.x + box.width * 0.5f + half;
              const float mark  = box.x + box.width * 0.5f;
              The same with the value named once:
              const float centre = box.x + box.width * 0.5f;
  where       --list prints each expression as written, spaces kept only between words, and how often
  method      this repository; tokens as libclang reads them, so a different spelling of the same value is not found.

restating comments   comments the code already says                                              lower is better
  asks        which comments only repeat the names on the next lines, so a reader reads everything twice?
  score       1 per comment of two or more content words, 60 % of which appear in identifiers within the next 5 lines
  rises when  a comment is added that says what the code's names already say
  ignore when the comment names a unit, a range, an empty value's meaning or a source (CLAUDE.md §2 allows these)
  do          delete it
  example     // the sample count                               // content words: sample, count
              const std::size_t sampleCount = input.size();     // both are in sampleCount: 2 of 2, restating
              The comment adds nothing the name does not say; delete it.
              // Heckbert's nice-number rule                    // content words: heckbert, nice, number, rule
              const double step = niceStep(range / kTickCount); // only nice is in niceStep: 1 of 4, not restating
              This one names the source of the algorithm, which no identifier holds; keep it.
  where       --list prints each with its file and text; --compare names those added or removed
  method      CLAUDE.md §2, "a comment that restates the code MUST be deleted"; content words are longer than two letters and
              not "the", "and", "for", …; identifiers split at "_" and at a lower-to-upper case change (sampleCount -> sample,
              count); licence, NOLINT, clang-format, URL and "// namespace" comments are exempt. A lexical test: a paraphrase
              escapes it.

long lines   lines over 160 and over 250 characters                                              lower is better
  asks        which lines can a reader not see whole, and so usually hold more than one idea?
  score       1 per line under the source roots longer than 160, and apart longer than 250, characters; a UTF-8 character counts once
  rises when  a call grows arguments that are themselves computations
  do          split the line into named const locals, each line one idea
  example     a call whose arguments are themselves computations, 163 characters on one line:
              drawLabel(layout.region(slide.areas[areaIndex]).shrunk(theme.margin * scale), theme.colour(Colour::accent).withAlpha(fadeInAlpha), label.text, kLabelSize * scale);
              split into named values, and each name says what the piece is for:
              const Rect   where  = layout.region(slide.areas[areaIndex]).shrunk(theme.margin * scale);
              const Colour colour = theme.colour(Colour::accent).withAlpha(fadeInAlpha);
              drawLabel(where, colour, label.text, kLabelSize * scale);   // no long line
  where       --list prints each file's count; --compare names the files that gained or lost long lines
  method      CLAUDE.md §9, "a line that reaches a formatter's limit is split into named values"; .clang-format sets no
              practical limit (2000), so this is the only place line length is seen.

(4) naming and API consistency: devtools/naming-check
(5) hot paths, instructions per item, binary and .wasm size: devtools/code-metrics.sh --hot-paths
(6) compile time: not measured yet

Not measured: whether a name is right (naming-check checks its form, not its meaning); whether a test's ground truth
is independent of the code (review); behaviour (the tests, and code-metrics.sh --seeded-faults).

references
  [1] J. Lakos, Large-Scale C++ Software Design, Addison-Wesley, 1996.
  [2] G. A. Campbell, "Cognitive Complexity", SonarSource, 2017.
  [3] C. Muratori, "Semantic Compression", 2014.
)method";

constexpr std::size_t kRestatementLookahead = 5; // lines after a comment whose identifiers it is compared with
constexpr double      kRestatementShare     = 0.6;

struct FunctionDefinition {
    std::string name;
    std::string signature; // qualified, with parameter types: one key per overload, the same in every run
    std::string file;
    unsigned    firstLine        = 0;
    unsigned    lastLine         = 0;
    bool        reachedOtherwise = false; // a constructor, destructor, operator, virtual function or main: called without being named
};

struct Call {
    std::string callerUsr;
    std::string callerFile;
    std::string calleeUsr;
    std::string site;             // file:line:column, so a call in a header counts once, not once per including unit
    unsigned    nestingDepth = 0; // if, ?:, loop, switch, catch and lambda levels around the call inside its caller

    bool operator<(const Call& other) const { return std::tie(site, calleeUsr, callerUsr) < std::tie(other.site, other.calleeUsr, other.callerUsr); }
};

struct FileFigures {
    std::size_t                                   tokens       = 0;
    std::size_t                                   sloc         = 0;
    std::size_t                                   commentLines = 0;
    std::size_t                                   commentWords = 0;
    std::size_t                                   over160      = 0;
    std::size_t                                   over250      = 0;
    std::vector<std::pair<unsigned, std::string>> restatements;        // line, comment text
    std::map<std::string, std::size_t>            repeatedExpressions; // as written -> times, for those written kRepeatedTimes or more
};

/// where a finding lies and what it is; reports key it without the line, which shifts with every edit above it
struct Located {
    std::string file;
    unsigned    line = 0;
    std::string what; // "function: local", "Record::field", "function: opener…closer #n"

    [[nodiscard]] std::string key() const { return std::format("{}: {}", file, what); }
    auto                      operator<=>(const Located&) const = default;
};

struct Findings {
    std::map<std::string, FunctionDefinition>                              definitions; // key: unified symbol resolution (USR)
    std::set<Call>                                                         calls;
    std::set<std::pair<std::string, std::string>>                          includes; // includer, included
    std::map<std::string, FileFigures>                                     files;
    std::map<Located, std::size_t>                                         multiStatementLocals;      // -> statement count
    std::map<Located, std::pair<int, std::string>>                         unscopedPairs;             // -> weight, what adds to it
    std::map<std::string, std::map<std::string, std::vector<std::string>>> constantArguments;         // callee USR -> call site -> each argument's text, empty when not a constant
    std::map<std::string, std::vector<std::string>>                        parameterNames;            // callee USR -> its parameters' names
    std::map<std::string, Located>                                         fieldsDeclared;            // field USR -> where it is declared, for fields under the source roots
    std::set<std::string>                                                  fieldsRead;                // field USRs read anywhere in own code, tests included
    std::set<std::string>                                                  fieldNamesReadDependently; // member names read through a dependent type, counted as a read of every field of that name
    std::set<std::string>                                                  recordsCompared;           // record USRs whose defaulted == or <=> reads every field
    std::set<std::string>                                                  functionsReferenced;       // function USRs named anywhere in own code: called, passed or taken by address
    std::set<std::string>                                                  namesUnresolved;           // names a template uses without libclang resolving them: every own function of that name is reached
    std::map<std::string, std::set<unsigned>>                              nestedTernaries;           // function USR -> offsets of each ?: that holds another in the same expression
    std::map<std::string, std::set<std::string>>                           externalLibraries;         // translation unit under the source roots -> libraries it includes
    std::map<std::string, std::map<std::string, std::set<std::string>>>    ambientSources;            // function USR -> category -> what it reaches directly
    std::set<std::string>                                                  unitsWithParseErrors;
};

// what is measured: the files under a root and not under an excluded directory; a file with a `test` or `benchmarks`
// directory in its path below the repository is a test, every other one a source; `core` is the part whose ambient reach is counted apart
struct Scope {
    fs::path              repository;
    std::vector<fs::path> roots;
    std::vector<fs::path> excluded;
    fs::path              core;

    [[nodiscard]] static bool     isUnder(std::string_view path, const fs::path& directory) { return path.starts_with(directory.string() + "/"); }
    [[nodiscard]] const fs::path* rootOf(std::string_view path) const {
        if (std::ranges::any_of(excluded, [path](const fs::path& directory) { return isUnder(path, directory); })) {
            return nullptr;
        }
        const auto root = std::ranges::find_if(roots, [path](const fs::path& directory) { return isUnder(path, directory); });
        return root == roots.end() ? nullptr : &*root;
    }
    [[nodiscard]] bool own(std::string_view path) const { return rootOf(path) != nullptr; }
    [[nodiscard]] bool inTests(std::string_view path) const {
        return own(path) && std::ranges::any_of(fs::relative(fs::path{path}.parent_path(), repository), [](const fs::path& segment) { return segment == "test" || segment == "benchmarks"; });
    }
    [[nodiscard]] bool        inSource(std::string_view path) const { return own(path) && !inTests(path); }
    [[nodiscard]] bool        inCore(std::string_view path) const { return !core.empty() && isUnder(path, core); }
    [[nodiscard]] std::string relative(const std::string& path) const { return fs::relative(path, repository).string(); }
    // X.hpp and X.cpp are one component; the same stem under two roots (algorithm/fft, blocks/fourier/fft) is two
    [[nodiscard]] std::string componentOf(const std::string& path) const {
        const fs::path*   root = rootOf(path);
        const std::string stem = fs::path{path}.stem().string();
        return root == nullptr ? stem : std::format("{}/{}", fs::relative(*root, repository).string(), stem);
    }
};

std::vector<std::string> wordsOf(std::string_view text) {
    std::vector<std::string> words;
    std::string              word;
    const auto               endWord = [&] {
        if (word.size() > 2) {
            words.push_back(word);
        }
        word.clear();
    };
    for (std::size_t i = 0; i < text.size(); ++i) {
        const auto character     = static_cast<unsigned char>(text[i]);
        const bool camelBoundary = i > 0 && (std::isupper(character) != 0) && (std::islower(static_cast<unsigned char>(text[i - 1])) != 0);
        if ((std::isalpha(character) == 0) || camelBoundary) {
            endWord();
        }
        if (std::isalpha(character) != 0) {
            word.push_back(static_cast<char>(std::tolower(character)));
        }
    }
    endWord();
    return words;
}

bool isStopWord(std::string_view word) {
    static const std::set<std::string_view> kStopWords{"the", "and", "for", "with", "from", "are", "its", "this", "that", "these", "those", "not", "when", "then", "than", "into", "onto", "per", "each", "one", "two", "all", "any", "only", "also", "but", "via", "while", "which", "what", "who"};
    return kStopWords.contains(word);
}

bool isFunction(CXCursorKind kind) { return kind == CXCursor_FunctionDecl || kind == CXCursor_CXXMethod || kind == CXCursor_FunctionTemplate || kind == CXCursor_ConversionFunction || kind == CXCursor_Constructor || kind == CXCursor_Destructor; }

// an expression of this many tokens or more, written this many times or more in one file, is a value that wants a name
constexpr unsigned    kRepeatedTokens = 8U;
constexpr std::size_t kRepeatedTimes  = 3UZ;

// tokens as written, spaced only between two words, as `a + b` and `a+b` are one writing
std::string writtenText(CXTranslationUnit unit, const CXToken* tokens, unsigned tokenCount) {
    std::string text;
    const auto  isWord = [](CXTokenKind tokenKind) { return tokenKind == CXToken_Identifier || tokenKind == CXToken_Keyword || tokenKind == CXToken_Literal; };
    for (unsigned index = 0; index < tokenCount; ++index) {
        if (index > 0 && isWord(clang_getTokenKind(tokens[index - 1])) && isWord(clang_getTokenKind(tokens[index]))) {
            text += ' ';
        }
        text += taken(clang_getTokenSpelling(unit, tokens[index]));
    }
    return text;
}

struct ExpressionWalk {
    CXTranslationUnit                         unit;
    CXFile                                    file;
    std::map<std::string, std::set<unsigned>> offsetsByText; // a wrapper with the same extent is the same writing
};

CXChildVisitResult visitExpression(CXCursor cursor, CXCursor, CXClientData walkContext) {
    ExpressionWalk&     walk   = *static_cast<ExpressionWalk*>(walkContext);
    const CXSourceRange extent = clang_getCursorExtent(cursor);
    CXFile              file   = nullptr;
    unsigned            offset = 0;
    clang_getExpansionLocation(clang_getRangeStart(extent), &file, nullptr, nullptr, &offset);
    if (file == nullptr || clang_File_isEqual(file, walk.file) == 0) {
        return CXChildVisit_Continue;
    }
    const CXCursorKind kind = clang_getCursorKind(cursor);
    if (clang_isExpression(kind) == 0 || kind == CXCursor_LambdaExpr) {
        return CXChildVisit_Recurse;
    }
    const auto writtenInFile = [&](CXSourceLocation location) {
        unsigned spelled   = 0;
        unsigned expansion = 0;
        clang_getSpellingLocation(location, nullptr, nullptr, nullptr, &spelled);
        clang_getExpansionLocation(location, nullptr, nullptr, nullptr, &expansion);
        return spelled == expansion;
    };
    if (!writtenInFile(clang_getRangeStart(extent)) || !writtenInFile(clang_getRangeEnd(extent))) {
        return CXChildVisit_Recurse;
    }
    CXToken* tokens     = nullptr;
    unsigned tokenCount = 0;
    clang_tokenize(walk.unit, extent, &tokens, &tokenCount);
    // a chain of names (viewer.navigator.cursor.step) is a reference, not a value computed again
    const auto isNameChain = [&] {
        for (unsigned index = 0; index < tokenCount; ++index) {
            const CXTokenKind tokenKind = clang_getTokenKind(tokens[index]);
            const std::string spelling  = tokenKind == CXToken_Punctuation ? taken(clang_getTokenSpelling(walk.unit, tokens[index])) : std::string{};
            if (tokenKind != CXToken_Identifier && tokenKind != CXToken_Keyword && spelling != "." && spelling != "->" && spelling != "::") {
                return false;
            }
        }
        return true;
    };
    if (tokenCount >= kRepeatedTokens && !isNameChain()) {
        walk.offsetsByText[writtenText(walk.unit, tokens, tokenCount)].insert(offset);
    }
    clang_disposeTokens(walk.unit, tokens, tokenCount);
    return CXChildVisit_Recurse;
}

std::map<std::string, std::size_t> repeatedExpressionsIn(CXTranslationUnit unit, CXFile file) {
    ExpressionWalk walk{.unit = unit, .file = file, .offsetsByText = {}};
    clang_visitChildren(clang_getTranslationUnitCursor(unit), visitExpression, &walk);
    std::map<std::string, std::size_t> repeated;
    for (const auto& [text, offsets] : walk.offsetsByText) {
        if (offsets.size() >= kRepeatedTimes) {
            repeated.emplace(text, offsets.size());
        }
    }
    // a part repeated only as often as a larger expression holding it is that expression, not a value of its own
    std::erase_if(repeated, [&repeated](const auto& part) { return std::ranges::any_of(repeated, [&part](const auto& whole) { return whole.first.size() > part.first.size() && whole.first.contains(part.first) && whole.second >= part.second; }); });
    return repeated;
}

FileFigures figuresOf(CXTranslationUnit unit, CXFile file, const std::string& path) {
    FileFigures              figures;
    std::vector<std::string> sourceLines;
    std::ifstream            fileStream{path};
    for (std::string line; std::getline(fileStream, line);) {
        const auto characters = static_cast<std::size_t>(std::ranges::count_if(line, [](char byte) { return (static_cast<unsigned char>(byte) & 0xC0U) != 0x80U; }));
        figures.over160 += characters > 160 ? 1 : 0;
        figures.over250 += characters > 250 ? 1 : 0;
        sourceLines.push_back(std::move(line));
    }
    const auto    fileSize   = static_cast<unsigned>(fs::file_size(path));
    CXSourceRange fileRange  = clang_getRange(clang_getLocationForOffset(unit, file, 0), clang_getLocationForOffset(unit, file, fileSize));
    CXToken*      tokens     = nullptr;
    unsigned      tokenCount = 0;
    clang_tokenize(unit, fileRange, &tokens, &tokenCount);

    std::map<unsigned, std::set<std::string>> identifierWordsByLine;
    std::set<unsigned>                        codeLines;
    struct Comment {
        std::string text;
        unsigned    line;
    };
    std::vector<Comment> comments;
    for (unsigned i = 0; i < tokenCount; ++i) {
        const CXSourceRange tokenExtent = clang_getTokenExtent(unit, tokens[i]);
        unsigned            firstLine = 0, lastLine = 0;
        clang_getSpellingLocation(clang_getRangeStart(tokenExtent), nullptr, &firstLine, nullptr, nullptr);
        clang_getSpellingLocation(clang_getRangeEnd(tokenExtent), nullptr, &lastLine, nullptr, nullptr);
        std::string spelling = taken(clang_getTokenSpelling(unit, tokens[i]));
        if (clang_getTokenKind(tokens[i]) == CXToken_Comment) {
            figures.commentLines += lastLine - firstLine + 1;
            comments.push_back(Comment{.text = std::move(spelling), .line = firstLine});
            continue;
        }
        ++figures.tokens;
        for (unsigned line = firstLine; line <= lastLine; ++line) {
            codeLines.insert(line);
        }
        if (clang_getTokenKind(tokens[i]) == CXToken_Identifier) {
            for (std::string& word : wordsOf(spelling)) {
                identifierWordsByLine[firstLine].insert(std::move(word));
            }
        }
    }
    clang_disposeTokens(unit, tokens, tokenCount);
    figures.sloc = codeLines.size();

    for (const Comment& comment : comments) {
        std::size_t letterRun = 0;
        for (std::size_t i = 0; i <= comment.text.size(); ++i) {
            const bool isLetter = i < comment.text.size() && (std::isalpha(static_cast<unsigned char>(comment.text[i])) != 0);
            figures.commentWords += !isLetter && letterRun >= 2 ? 1 : 0;
            letterRun = isLetter ? letterRun + 1 : 0;
        }
        const bool            isExempt = comment.text.starts_with("// namespace") || std::ranges::any_of(std::array<std::string_view, 6>{"NOLINT", "clang-format", "http", "@source", "SPDX", "icen"}, [&](std::string_view marker) { return comment.text.contains(marker); });
        std::set<std::string> contentWords;
        for (std::string& word : wordsOf(comment.text)) {
            if (!isStopWord(word)) {
                contentWords.insert(std::move(word));
            }
        }
        if (isExempt || contentWords.size() < 2) {
            continue;
        }
        std::set<std::string> nearbyIdentifierWords;
        for (unsigned line = comment.line; line <= comment.line + kRestatementLookahead; ++line) {
            if (const auto found = identifierWordsByLine.find(line); found != identifierWordsByLine.end()) {
                nearbyIdentifierWords.insert(found->second.begin(), found->second.end());
            }
        }
        const auto sharedWordCount = std::ranges::count_if(contentWords, [&](const std::string& word) { return nearbyIdentifierWords.contains(word); });
        if (static_cast<double>(sharedWordCount) >= kRestatementShare * static_cast<double>(contentWords.size())) {
            std::string text = comment.text.substr(0, 100);
            std::ranges::replace(text, '\n', ' ');
            figures.restatements.emplace_back(comment.line, std::move(text));
        }
    }
    return figures;
}

struct Walk {
    Findings*    findings;
    const Scope* scope;
    std::string  functionUsr;
    std::string  functionFile;
    unsigned     nestingDepth = 0;
};

// what a call reaches beyond its arguments; write-only logging (gr::log, printing to stdout or stderr) is not a source
std::string_view ambientCategoryOfCallee(const std::string& calleeUsr) {
    const auto isCFunction = [&calleeUsr](std::string_view prefix) { return calleeUsr.starts_with(std::format("c:@F@{}", prefix)); };
    const bool isOpenGl    = calleeUsr.starts_with("c:@F@gl") && calleeUsr.size() > 7 && (std::isupper(static_cast<unsigned char>(calleeUsr[7])) != 0);
    if (calleeUsr.contains("@N@ImGui@") || calleeUsr.contains("@N@ImPlot@") || calleeUsr.contains("@N@NodeEditor@") || isCFunction("SDL_") || isCFunction("emscripten_") || isOpenGl) {
        return "drawing context";
    }
    if (calleeUsr.contains("@N@chrono@") && calleeUsr.contains("@F@now#")) {
        return "clock";
    }
    if (isCFunction("getenv") || calleeUsr.contains("@N@std@F@getenv#")) {
        return "environment";
    }
    const bool touchesFileSystem = calleeUsr.contains("@N@filesystem@") && !calleeUsr.contains("@S@path@");
    const bool isFileStream      = calleeUsr.contains("@S@basic_ifstream") || calleeUsr.contains("@S@basic_ofstream") || calleeUsr.contains("@S@basic_fstream");
    if (touchesFileSystem || isFileStream || calleeUsr.contains("@N@cpr@") || isCFunction("fopen")) {
        return "io";
    }
    return {};
}

std::string_view ambientCategoryOfVariable(CXCursor variable) {
    if (clang_getCursorKind(variable) != CXCursor_VarDecl || clang_isConstQualifiedType(clang_getCursorType(variable)) != 0) {
        return {};
    }
    const CXCursorKind parentKind       = clang_getCursorKind(clang_getCursorSemanticParent(variable));
    const bool         atNamespaceScope = parentKind == CXCursor_Namespace || parentKind == CXCursor_TranslationUnit || parentKind == CXCursor_LinkageSpec;
    const bool         isStatic         = clang_Cursor_getStorageClass(variable) == CX_SC_Static || clang_getCursorTLSKind(variable) != CXTLS_None;
    if (!atNamespaceScope && !isStatic) {
        return {};
    }
    const std::string name = taken(clang_getCursorSpelling(variable));
    if (std::ranges::contains(std::array<std::string_view, 6>{"stdout", "stderr", "cout", "cerr", "clog", "wcerr"}, name)) {
        return {};
    }
    return name.starts_with("glad_gl") ? "drawing context" : "global";
}

bool isNesting(CXCursorKind kind) { return kind == CXCursor_IfStmt || kind == CXCursor_ForStmt || kind == CXCursor_CXXForRangeStmt || kind == CXCursor_WhileStmt || kind == CXCursor_DoStmt || kind == CXCursor_SwitchStmt || kind == CXCursor_CXXCatchStmt || kind == CXCursor_ConditionalOperator || kind == CXCursor_LambdaExpr; }

struct Assignments {
    CXCursor                                                             function; // a function, a function template or a lambda
    std::size_t                                                          statement = 0UZ;
    std::map<std::string, std::pair<std::string, std::set<std::size_t>>> byLocal; // local's USR -> name, top-level statements
};

// the local a write lands in: `x = …`, `x.field += …`, `++x[i]` all change `x`
CXCursor writtenLocal(CXCursor target) {
    while ((clang_Cursor_isNull(target) == 0) && clang_getCursorKind(target) != CXCursor_DeclRefExpr) {
        const CXCursorKind kind = clang_getCursorKind(target);
        if (kind != CXCursor_UnexposedExpr && kind != CXCursor_ParenExpr && kind != CXCursor_MemberRefExpr && kind != CXCursor_ArraySubscriptExpr) {
            return clang_getNullCursor();
        }
        target = firstChildOf(target);
    }
    return (clang_Cursor_isNull(target) != 0) ? target : clang_getCursorReferenced(target);
}

// a nested lambda is counted on its own: its assignments fall in one top-level statement of `scope`
bool isLocalOf(CXCursor local, CXCursor scope) {
    const CXCursorKind kind = clang_getCursorKind(local);
    if (kind != CXCursor_VarDecl && kind != CXCursor_ParmDecl) {
        return false;
    }
    const CXCursorKind parentKind = clang_getCursorKind(clang_getCursorSemanticParent(local));
    if (parentKind == CXCursor_Namespace || parentKind == CXCursor_TranslationUnit || parentKind == CXCursor_StructDecl || parentKind == CXCursor_ClassDecl) {
        return false;
    }
    CXFile              localFile   = nullptr;
    CXFile              scopeFile   = nullptr;
    unsigned            localOffset = 0;
    unsigned            scopeStart  = 0;
    unsigned            scopeEnd    = 0;
    const CXSourceRange scopeExtent = clang_getCursorExtent(scope);
    clang_getExpansionLocation(clang_getCursorLocation(local), &localFile, nullptr, nullptr, &localOffset);
    clang_getExpansionLocation(clang_getRangeStart(scopeExtent), &scopeFile, nullptr, nullptr, &scopeStart);
    clang_getExpansionLocation(clang_getRangeEnd(scopeExtent), nullptr, nullptr, nullptr, &scopeEnd);
    return clang_File_isEqual(localFile, scopeFile) != 0 && localOffset >= scopeStart && localOffset <= scopeEnd;
}

CXChildVisitResult visitAssignment(CXCursor cursor, CXCursor, CXClientData assignmentsContext) {
    Assignments&               assignments    = *static_cast<Assignments*>(assignmentsContext);
    const CXCursorKind         kind           = clang_getCursorKind(cursor);
    const CXBinaryOperatorKind binaryOperator = kind == CXCursor_BinaryOperator || kind == CXCursor_CompoundAssignOperator ? clang_getCursorBinaryOperatorKind(cursor) : CXBinaryOperator_Invalid;
    const CXUnaryOperatorKind  unaryOperator  = kind == CXCursor_UnaryOperator ? clang_getCursorUnaryOperatorKind(cursor) : CXUnaryOperator_Invalid;
    const bool                 isWrite        = (binaryOperator >= CXBinaryOperator_Assign && binaryOperator <= CXBinaryOperator_OrAssign) || (unaryOperator >= CXUnaryOperator_PostInc && unaryOperator <= CXUnaryOperator_PreDec);
    if (isWrite) {
        const CXCursor writtenVariable = writtenLocal(firstChildOf(cursor));
        if ((clang_Cursor_isNull(writtenVariable) == 0) && isLocalOf(writtenVariable, assignments.function)) {
            auto& [name, statements] = assignments.byLocal[taken(clang_getCursorUSR(writtenVariable))];
            name                     = taken(clang_getCursorSpelling(writtenVariable));
            statements.insert(assignments.statement);
        }
    }
    return CXChildVisit_Recurse;
}

void countMultiStatementLocals(CXCursor function, const std::string& file, unsigned line, Findings& findings) {
    CXCursor functionBody = clang_getNullCursor();
    clang_visitChildren(
        function,
        [](CXCursor child, CXCursor, CXClientData bodySlot) {
            if (clang_getCursorKind(child) == CXCursor_CompoundStmt) {
                *static_cast<CXCursor*>(bodySlot) = child;
                return CXChildVisit_Break;
            }
            return CXChildVisit_Continue;
        },
        &functionBody);
    if (clang_Cursor_isNull(functionBody) != 0) {
        return;
    }
    Assignments assignments{.function = function, .statement = 0UZ, .byLocal = {}};
    clang_visitChildren(
        functionBody,
        [](CXCursor statement, CXCursor, CXClientData assignmentsContext) {
            Assignments& counting = *static_cast<Assignments*>(assignmentsContext);
            visitAssignment(statement, statement, assignmentsContext);
            clang_visitChildren(statement, visitAssignment, assignmentsContext);
            ++counting.statement;
            return CXChildVisit_Continue;
        },
        &assignments);
    const std::string functionName = taken(clang_getCursorSpelling(function));
    for (const auto& [localUsr, nameAndStatements] : assignments.byLocal) {
        if (nameAndStatements.second.size() > 1UZ) {
            findings.multiStatementLocals.try_emplace(Located{.file = file, .line = line, .what = std::format("{}: {}", functionName, nameAndStatements.first)}, nameAndStatements.second.size());
        }
    }
}

constexpr std::array<std::pair<std::string_view, std::string_view>, 9> kPairPrefixes{{{"Push", "Pop"}, {"Begin", "End"}, {"glGen", "glDelete"}, {"Lock", "Unlock"}, {"lock", "unlock"}, {"Acquire", "Release"}, {"acquire", "release"}, {"open", "close"}, {"malloc", "free"}}};

struct PairEvent {
    unsigned    offset;
    std::string name; // the call's spelling; empty for an exit
    std::string key;  // which pair and the rest of the name; an opener and its closer share it
    bool        opens;
    bool        exit;      // return, throw, break or continue
    unsigned    loopStart; // where a break's or continue's own loop or switch starts; 0 for return and throw, which leave every scope
    std::string branch;    // the branches of if, switch and loop bodies it sits in, as a path
};

struct PairWalk {
    std::vector<PairEvent> events;
    std::string            branch;
    unsigned               loopStart = 0;
};

unsigned offsetOf(CXCursor cursor) {
    unsigned offset = 0;
    clang_getExpansionLocation(clang_getCursorLocation(cursor), nullptr, nullptr, nullptr, &offset);
    return offset;
}

void walkPairs(CXCursor parent, PairWalk& walk) {
    std::vector<CXCursor> children;
    clang_visitChildren(
        parent,
        [](CXCursor child, CXCursor, CXClientData into) {
            static_cast<std::vector<CXCursor>*>(into)->push_back(child);
            return CXChildVisit_Continue;
        },
        &children);
    const CXCursorKind parentKind = clang_getCursorKind(parent);
    const bool         loops      = parentKind == CXCursor_ForStmt || parentKind == CXCursor_WhileStmt || parentKind == CXCursor_DoStmt || parentKind == CXCursor_CXXForRangeStmt || parentKind == CXCursor_SwitchStmt;
    const bool         branches   = loops || parentKind == CXCursor_IfStmt;
    for (std::size_t index = 0UZ; index < children.size(); ++index) {
        const CXCursor     child = children[index];
        const CXCursorKind kind  = clang_getCursorKind(child);
        if (kind == CXCursor_LambdaExpr) {
            continue; // a lambda's returns leave the lambda, and its pairs are counted with it
        }
        const std::string branchBefore = walk.branch;
        const unsigned    loopBefore   = walk.loopStart;
        if (branches && clang_isStatement(kind) != 0) { // a condition always runs; only a body is a branch
            walk.branch += std::format("/{}:{}", offsetOf(parent), index);
        }
        if (loops) {
            walk.loopStart = offsetOf(parent);
        }
        if (kind == CXCursor_CallExpr) {
            const std::string      spelling   = taken(clang_getCursorSpelling(child));
            const std::size_t      underscore = spelling.rfind('_');
            const bool             prefixed   = underscore != std::string::npos && underscore + 1UZ < spelling.size() && std::isupper(static_cast<unsigned char>(spelling[underscore + 1UZ])) != 0;
            const std::string_view name       = prefixed ? std::string_view{spelling}.substr(underscore + 1UZ) : std::string_view{spelling}; // SDL_LockMutex is LockMutex; push_back stays itself
            for (std::size_t pair = 0UZ; pair < kPairPrefixes.size(); ++pair) {
                const auto& [opener, closer] = kPairPrefixes[pair];
                const bool opens             = name.starts_with(opener);
                if (opens || name.starts_with(closer)) {
                    const std::string_view rest = name.substr(opens ? opener.size() : closer.size());
                    walk.events.push_back(PairEvent{.offset = offsetOf(child), .name = spelling, .key = std::format("{}:{}", pair, rest), .opens = opens, .exit = false, .loopStart = 0U, .branch = walk.branch});
                    break;
                }
            }
        }
        if (kind == CXCursor_ReturnStmt || kind == CXCursor_CXXThrowExpr || kind == CXCursor_BreakStmt || kind == CXCursor_ContinueStmt) {
            const bool leavesEverything = kind == CXCursor_ReturnStmt || kind == CXCursor_CXXThrowExpr;
            walk.events.push_back(PairEvent{.offset = offsetOf(child), .name = {}, .key = {}, .opens = false, .exit = true, .loopStart = leavesEverything ? 0U : walk.loopStart, .branch = walk.branch});
        }
        walkPairs(child, walk);
        walk.branch    = branchBefore;
        walk.loopStart = loopBefore;
    }
}

void countUnscopedPairs(CXCursor function, const std::string& file, unsigned line, Findings& findings) {
    PairWalk walk;
    walkPairs(function, walk);
    std::ranges::sort(walk.events, {}, &PairEvent::offset);
    std::map<std::string, std::vector<const PairEvent*>> opened;
    std::map<std::string, int>                           ordinal;
    const std::string                                    functionName = taken(clang_getCursorSpelling(function));
    for (const PairEvent& event : walk.events) {
        if (event.exit) {
            continue;
        }
        std::vector<const PairEvent*>& stack = opened[event.key];
        if (event.opens) {
            stack.push_back(&event);
            continue;
        }
        if (stack.empty()) {
            continue; // closes what another function opened: a scope object's destructor, or a caller's push
        }
        const PairEvent& opener = *stack.back();
        stack.pop_back();
        // an exit between them skips the closing, unless it is a break or continue of a loop that lies wholly inside the pair
        const auto        exits = std::ranges::count_if(walk.events, [&](const PairEvent& between) { return between.exit && between.offset > opener.offset && between.offset < event.offset && between.loopStart < opener.offset; });
        const bool        split = opener.branch != event.branch;
        const std::string pair  = std::format("{}…{}", opener.name, event.name);
        const std::string what  = std::format("{} exit{} between{}", exits, exits == 1 ? "" : "s", split ? ", opened and closed in different branches" : "");
        findings.unscopedPairs.try_emplace(Located{.file = file, .line = line, .what = std::format("{}: {} #{}", functionName, pair, ++ordinal[pair])}, 1 + static_cast<int>(exits) + (split ? 1 : 0), what);
    }
}

// a literal, a kConstant or a qualified enumerator (Selection::all): the same text means the same value at every call
std::string constantArgumentText(CXCursor argument) {
    const CXTranslationUnit unit       = clang_Cursor_getTranslationUnit(argument);
    CXToken*                tokens     = nullptr;
    unsigned                tokenCount = 0;
    clang_tokenize(unit, clang_getCursorExtent(argument), &tokens, &tokenCount);
    bool constant = tokenCount > 0;
    for (unsigned index = 0; index < tokenCount && constant; ++index) {
        if (clang_getTokenKind(tokens[index]) != CXToken_Identifier) {
            continue;
        }
        const std::string name      = taken(clang_getTokenSpelling(unit, tokens[index]));
        const bool        qualified = (index > 0 && taken(clang_getTokenSpelling(unit, tokens[index - 1])) == "::") || (index + 1 < tokenCount && taken(clang_getTokenSpelling(unit, tokens[index + 1])) == "::");
        constant                    = qualified || (name.size() > 1 && name[0] == 'k' && std::isupper(static_cast<unsigned char>(name[1])) != 0);
    }
    std::string text = constant ? writtenText(unit, tokens, tokenCount) : std::string{};
    clang_disposeTokens(unit, tokens, tokenCount);
    return text;
}

// a field written only by `=`, a designated initialiser or a constructor's member initialiser is never read; += and ++ read it too
void noteFieldUse(CXCursor cursor, CXCursor parent, Findings& findings) {
    const CXCursorKind kind      = clang_getCursorKind(cursor);
    const CXCursor     field     = clang_getCursorReferenced(cursor);
    const bool         dependent = kind == CXCursor_MemberRefExpr && clang_isInvalid(clang_getCursorKind(field)) != 0;
    if (clang_getCursorKind(field) != CXCursor_FieldDecl && !dependent) {
        return;
    }
    const CXCursorKind parentKind = clang_getCursorKind(parent);
    const auto         startOf    = [](CXCursor of) {
        unsigned offset = 0;
        clang_getExpansionLocation(clang_getRangeStart(clang_getCursorExtent(of)), nullptr, nullptr, nullptr, &offset);
        return offset;
    };
    const bool assignedTo  = parentKind == CXCursor_BinaryOperator && clang_getCursorBinaryOperatorKind(parent) == CXBinaryOperator_Assign && startOf(cursor) == startOf(parent);
    const bool initialised = kind == CXCursor_MemberRef;
    if (assignedTo || initialised) {
        return;
    }
    if (dependent) {
        CXToken*          tokens     = nullptr;
        unsigned          tokenCount = 0;
        CXTranslationUnit unit       = clang_Cursor_getTranslationUnit(cursor);
        clang_tokenize(unit, clang_getCursorExtent(cursor), &tokens, &tokenCount);
        if (tokenCount > 0) {
            findings.fieldNamesReadDependently.insert(taken(clang_getTokenSpelling(unit, tokens[tokenCount - 1])));
        }
        clang_disposeTokens(unit, tokens, tokenCount);
    } else {
        findings.fieldsRead.insert(taken(clang_getCursorUSR(field)));
    }
}

const std::set<std::string_view> kBlockHooks{"processOne", "processBulk", "processEpilogue", "processMessages", "start", "stop", "pause", "resume", "reset", "settingsChanged", "work"};

CXChildVisitResult visitCursor(CXCursor cursor, CXCursor, CXClientData walkContext);

// the children of an if, ?:, loop, switch, catch or lambda one nesting level deeper; an `else if` stays at its `if`'s level
void visitNested(CXCursor nesting, const Walk& walk) {
    std::vector<CXCursor> children;
    clang_visitChildren(
        nesting,
        [](CXCursor child, CXCursor, CXClientData childrenSlot) {
            static_cast<std::vector<CXCursor>*>(childrenSlot)->push_back(child);
            return CXChildVisit_Continue;
        },
        &children);
    const bool hasElseIf     = clang_getCursorKind(nesting) == CXCursor_IfStmt && children.size() >= 3 && clang_getCursorKind(children.back()) == CXCursor_IfStmt;
    Walk       sameLevelWalk = walk;
    Walk       nestedWalk    = walk;
    ++nestedWalk.nestingDepth;
    for (std::size_t childIndex = 0; childIndex < children.size(); ++childIndex) {
        Walk& childWalk = hasElseIf && childIndex + 1 == children.size() ? sameLevelWalk : nestedWalk;
        if (visitCursor(children[childIndex], nesting, &childWalk) == CXChildVisit_Recurse) {
            clang_visitChildren(children[childIndex], visitCursor, &childWalk);
        }
    }
}

CXChildVisitResult visitCursor(CXCursor cursor, CXCursor syntacticParent, CXClientData walkContext) {
    Walk&             walk = *static_cast<Walk*>(walkContext);
    unsigned          line = 0;
    const std::string file = fileOf(clang_getCursorLocation(cursor), &line);
    if (file.empty()) {
        return CXChildVisit_Recurse; // an implicit node (a conversion operator's call) has no place, but what it converts does
    }
    if (!walk.scope->own(file)) {
        return clang_getCursorKind(cursor) == CXCursor_Namespace || clang_getCursorKind(cursor) == CXCursor_LinkageSpec ? CXChildVisit_Recurse : CXChildVisit_Continue;
    }
    const CXCursorKind kind = clang_getCursorKind(cursor);
    if (kind == CXCursor_MemberRefExpr || kind == CXCursor_DeclRefExpr || kind == CXCursor_MemberRef) {
        noteFieldUse(cursor, syntacticParent, *walk.findings);
        CXCursor named = clang_getCursorReferenced(cursor);
        if (const CXCursor specialisedTemplate = clang_getSpecializedCursorTemplate(named); clang_Cursor_isNull(specialisedTemplate) == 0) {
            named = specialisedTemplate;
        }
        if (isFunction(clang_getCursorKind(named))) {
            walk.findings->functionsReferenced.insert(taken(clang_getCursorUSR(clang_getCanonicalCursor(named))));
        } else if ((clang_Cursor_isNull(named) != 0) || clang_getCursorKind(named) == CXCursor_OverloadedDeclRef) {
            walk.findings->namesUnresolved.insert(taken(clang_getCursorSpelling(cursor)));
        }
    }
    if (kind == CXCursor_OverloadedDeclRef) {
        walk.findings->namesUnresolved.insert(taken(clang_getCursorSpelling(cursor)));
    }
    if (kind == CXCursor_FieldDecl && walk.scope->inSource(file)) {
        const CXCursor record     = clang_getCursorSemanticParent(cursor);
        std::string    recordName = taken(clang_getCursorSpelling(record));
        if (clang_Cursor_isAnonymous(record) != 0) { // spelled with its file and line, which move: named after the member it declares instead
            struct Search {
                CXCursor    record;
                std::string member;
            } search{.record = record, .member = {}};
            clang_visitChildren(
                clang_getCursorSemanticParent(record),
                [](CXCursor child, CXCursor, CXClientData searchSlot) {
                    Search& found = *static_cast<Search*>(searchSlot);
                    if (clang_getCursorKind(child) == CXCursor_FieldDecl && clang_equalCursors(clang_getTypeDeclaration(clang_getCursorType(child)), found.record) != 0) {
                        found.member = taken(clang_getCursorSpelling(child));
                        return CXChildVisit_Break;
                    }
                    return CXChildVisit_Continue;
                },
                &search);
            recordName = std::format("{}::{}", taken(clang_getCursorSpelling(clang_getCursorSemanticParent(record))), search.member.empty() ? "(unnamed)" : search.member);
        }
        walk.findings->fieldsDeclared.try_emplace(taken(clang_getCursorUSR(cursor)), Located{.file = walk.scope->relative(file), .line = line, .what = std::format("{}::{}", recordName, taken(clang_getCursorSpelling(cursor)))});
    }
    if (kind == CXCursor_CXXMethod && clang_CXXMethod_isDefaulted(cursor) != 0) {
        if (const std::string name = taken(clang_getCursorSpelling(cursor)); name == "operator==" || name == "operator<=>") {
            walk.findings->recordsCompared.insert(taken(clang_getCursorUSR(clang_getCursorSemanticParent(cursor))));
        }
    }
    if (isFunction(kind) && (clang_isCursorDefinition(cursor) != 0u)) {
        const std::string definitionUsr = taken(clang_getCursorUSR(cursor));
        if (walk.scope->inSource(file)) {
            unsigned lastLine = 0;
            clang_getExpansionLocation(clang_getRangeEnd(clang_getCursorExtent(cursor)), nullptr, &lastLine, nullptr, nullptr);
            std::string signature = taken(clang_getCursorDisplayName(cursor));
            for (CXCursor parent = clang_getCursorSemanticParent(cursor); clang_getCursorKind(parent) != CXCursor_TranslationUnit && (clang_Cursor_isNull(parent) == 0); parent = clang_getCursorSemanticParent(parent)) {
                const std::string parentName = taken(clang_getCursorSpelling(parent));
                signature                    = std::format("{}::{}", parentName.empty() ? "(anonymous)" : parentName, signature);
            }
            const std::string name = taken(clang_getCursorSpelling(cursor));
            // a framework calls the members of a class derived from its base (GR4's processBulk), and a macro writes functions nobody names
            const auto derivedFrom = [&] {
                struct BaseSearch {
                    const Scope* scope;
                    bool         elsewhere;
                    bool         blockBase;
                } search{.scope = walk.scope, .elsewhere = false, .blockBase = false};
                clang_visitChildren(
                    clang_getCursorSemanticParent(cursor),
                    [](CXCursor child, CXCursor, CXClientData searchContext) {
                        BaseSearch& found = *static_cast<BaseSearch*>(searchContext);
                        if (clang_getCursorKind(child) == CXCursor_CXXBaseSpecifier) {
                            unsigned          ignored  = 0;
                            const CXCursor    base     = clang_getCursorReferenced(child); // a dependent base (gr::Block<Block<T>>) has a declaration, not a definition
                            const std::string baseFile = fileOf(clang_getCursorLocation(clang_Cursor_isNull(base) ? clang_getTypeDeclaration(clang_getCursorType(child)) : base), &ignored);
                            found.elsewhere            = found.elsewhere || (!baseFile.empty() && !found.scope->own(baseFile));
                            const std::string baseType = taken(clang_getTypeSpelling(clang_getCursorType(child)));
                            found.blockBase            = found.blockBase || baseType.starts_with("Block<") || baseType.starts_with("gr::Block<");
                        }
                        return CXChildVisit_Continue;
                    },
                    &search);
                return search;
            };
            CXFile spelledIn  = nullptr;
            CXFile expandedIn = nullptr;
            clang_getSpellingLocation(clang_getCursorLocation(cursor), &spelledIn, nullptr, nullptr, nullptr);
            clang_getExpansionLocation(clang_getCursorLocation(cursor), &expandedIn, nullptr, nullptr, nullptr);
            const bool         fromMacro        = spelledIn != nullptr && expandedIn != nullptr && clang_File_isEqual(spelledIn, expandedIn) == 0;
            const bool         isMethod         = kind == CXCursor_CXXMethod || kind == CXCursor_FunctionTemplate;
            const bool         reachedOtherwise = kind == CXCursor_Constructor || kind == CXCursor_Destructor || kind == CXCursor_ConversionFunction || name.starts_with("operator") || name == "main" || clang_CXXMethod_isVirtual(cursor) != 0 || fromMacro || (isMethod && derivedFrom().elsewhere) || (isMethod && kBlockHooks.contains(name) && derivedFrom().blockBase);
            FunctionDefinition definition{.name = name, .signature = std::move(signature), .file = file, .firstLine = line, .lastLine = lastLine, .reachedOtherwise = reachedOtherwise};
            walk.findings->definitions.try_emplace(definitionUsr, std::move(definition));
            countMultiStatementLocals(cursor, walk.scope->relative(file), line, *walk.findings);
            countUnscopedPairs(cursor, walk.scope->relative(file), line, *walk.findings);
        }
        Walk functionWalk{.findings = walk.findings, .scope = walk.scope, .functionUsr = definitionUsr, .functionFile = file, .nestingDepth = 0};
        clang_visitChildren(cursor, visitCursor, &functionWalk);
        return CXChildVisit_Continue;
    }
    if (kind == CXCursor_LambdaExpr && walk.scope->inSource(file)) {
        countMultiStatementLocals(cursor, walk.scope->relative(file), line, *walk.findings);
        countUnscopedPairs(cursor, walk.scope->relative(file), line, *walk.findings);
    }
    const bool insideTestCase = walk.functionUsr.empty() && walk.scope->inTests(file); // Boost.UT cases are lambdas in a global initialiser
    if (kind == CXCursor_CallExpr && (!walk.functionUsr.empty() || insideTestCase)) {
        CXCursor callee = clang_getCursorReferenced(cursor);
        if (const CXCursor specialisedTemplate = clang_getSpecializedCursorTemplate(callee); clang_Cursor_isNull(specialisedTemplate) == 0) {
            callee = specialisedTemplate;
        }
        if ((clang_Cursor_isNull(callee) == 0) && isFunction(clang_getCursorKind(callee))) {
            const std::string calleeUsr = taken(clang_getCursorUSR(clang_getCanonicalCursor(callee)));
            unsigned          column    = 0;
            clang_getExpansionLocation(clang_getCursorLocation(cursor), nullptr, nullptr, &column, nullptr);
            const std::string site = std::format("{}:{}:{}", file, line, column);
            walk.findings->calls.insert(insideTestCase ? Call{.callerUsr = file, .callerFile = file, .calleeUsr = calleeUsr, .site = site, .nestingDepth = 0} : Call{.callerUsr = walk.functionUsr, .callerFile = walk.functionFile, .calleeUsr = calleeUsr, .site = site, .nestingDepth = walk.nestingDepth});
            if (const int argumentCount = clang_Cursor_getNumArguments(cursor); argumentCount > 0 && !taken(clang_getCursorSpelling(callee)).starts_with("operator")) {
                std::vector<std::string>& arguments = walk.findings->constantArguments[calleeUsr][site];
                for (int index = 0; index < argumentCount; ++index) {
                    arguments.push_back(constantArgumentText(clang_Cursor_getArgument(cursor, static_cast<unsigned>(index))));
                }
                if (std::vector<std::string>& names = walk.findings->parameterNames[calleeUsr]; names.empty()) {
                    for (int index = 0; index < clang_Cursor_getNumArguments(callee); ++index) {
                        names.push_back(taken(clang_getCursorSpelling(clang_Cursor_getArgument(callee, static_cast<unsigned>(index)))));
                    }
                }
            }
        }
    }
    const bool insideSourceFunction = !walk.functionUsr.empty() && walk.scope->inSource(walk.functionFile);
    if (insideSourceFunction && (kind == CXCursor_CallExpr || kind == CXCursor_DeclRefExpr || kind == CXCursor_MemberRefExpr)) {
        // a call can reach libclang as a bare reference (a static member returning by value is an unexposed expression)
        const CXCursor         referenced     = clang_getCursorReferenced(cursor);
        const CXCursorKind     referencedKind = clang_getCursorKind(referenced);
        const bool             isCallable     = isFunction(referencedKind);
        const std::string_view category       = isCallable ? ambientCategoryOfCallee(taken(clang_getCursorUSR(clang_getCanonicalCursor(referenced)))) : ambientCategoryOfVariable(referenced);
        if (!category.empty()) {
            walk.findings->ambientSources[walk.functionUsr][std::string{category}].insert(taken(clang_getCursorSpelling(referenced)));
        }
    }
    if (insideSourceFunction && kind == CXCursor_ConditionalOperator) {
        bool holdsAnother = false;
        clang_visitChildren(
            cursor,
            [](CXCursor child, CXCursor, CXClientData found) {
                const CXCursorKind childKind = clang_getCursorKind(child);
                if (childKind == CXCursor_LambdaExpr) {
                    return CXChildVisit_Continue; // a lambda's body is an expression of its own
                }
                *static_cast<bool*>(found) = *static_cast<bool*>(found) || childKind == CXCursor_ConditionalOperator;
                return CXChildVisit_Recurse;
            },
            &holdsAnother);
        if (holdsAnother) {
            walk.findings->nestedTernaries[walk.functionUsr].insert(offsetOf(cursor));
        }
    }
    if (insideSourceFunction && isNesting(kind)) {
        visitNested(cursor, walk);
        return CXChildVisit_Continue;
    }
    return CXChildVisit_Recurse;
}

struct InclusionWalk {
    Findings*                                   findings;
    const Scope*                                scope;
    std::string                                 unitSource;
    std::vector<std::pair<std::string, CXFile>> ownFiles;
};

std::string libraryOf(const std::string& path) {
    if (const std::size_t deps = path.find("/_deps/"); deps != std::string::npos) {
        std::string name = path.substr(deps + 7UZ, path.find('/', deps + 7UZ) - deps - 7UZ);
        for (const std::string_view suffix : {"-src", "-build", "-subbuild"}) {
            if (name.ends_with(suffix)) {
                name.resize(name.size() - suffix.size());
            }
        }
        return name;
    }
    if (path.contains("/c++/") || path.contains("/clang/") || path.contains("/gcc/")) {
        return "std";
    }
    return path.starts_with("/usr/") ? "system" : "other";
}

void visitInclusion(CXFile included, CXSourceLocation* stack, unsigned depth, CXClientData walkContext) {
    InclusionWalk&    walk         = *static_cast<InclusionWalk*>(walkContext);
    const std::string includedPath = fs::weakly_canonical(taken(clang_getFileName(included))).string();
    if (!walk.scope->own(includedPath)) {
        if (walk.scope->inSource(walk.unitSource)) {
            walk.findings->externalLibraries[walk.unitSource].insert(libraryOf(includedPath));
        }
        return;
    }
    walk.ownFiles.emplace_back(includedPath, included);
    if (depth > 0) {
        if (const std::string includer = fileOf(stack[0]); walk.scope->own(includer)) {
            walk.findings->includes.emplace(includer, includedPath);
        }
    }
}

Findings analyse(CXTranslationUnit unit, CXCompileCommand command, const Scope& scope, std::mutex& claimLock, std::set<std::string>& claimedFiles) {
    Findings      findings;
    InclusionWalk inclusions{.findings = &findings, .scope = &scope, .unitSource = fs::weakly_canonical(taken(clang_CompileCommand_getFilename(command))).string(), .ownFiles = {}};
    clang_getInclusions(unit, visitInclusion, &inclusions);
    for (const auto& [path, file] : inclusions.ownFiles) {
        bool claimedHere = false;
        {
            const std::scoped_lock lock{claimLock};
            claimedHere = scope.inSource(path) && claimedFiles.insert(path).second;
        }
        if (claimedHere) {
            FileFigures figures         = figuresOf(unit, file, path);
            figures.repeatedExpressions = repeatedExpressionsIn(unit, file);
            findings.files.emplace(path, std::move(figures));
        }
    }
    for (unsigned diagnosticIndex = 0; diagnosticIndex < clang_getNumDiagnostics(unit); ++diagnosticIndex) {
        CXDiagnostic diagnostic = clang_getDiagnostic(unit, diagnosticIndex);
        if (clang_getDiagnosticSeverity(diagnostic) >= CXDiagnostic_Error) {
            findings.unitsWithParseErrors.insert(inclusions.unitSource);
        }
        clang_disposeDiagnostic(diagnostic);
    }
    Walk walk{.findings = &findings, .scope = &scope, .functionUsr = {}, .functionFile = {}, .nestingDepth = 0};
    clang_visitChildren(clang_getTranslationUnitCursor(unit), visitCursor, &walk);
    return findings;
}

/// a null stream only collects (the summary, table and comparison modes)
template<typename... TArguments>
void say(std::FILE* out, std::format_string<TArguments...> text, TArguments&&... arguments) {
    if (out != nullptr) {
        std::println(out, text, std::forward<TArguments>(arguments)...);
    }
}

using MeasuredFigures = std::map<std::string, double, std::less<>>; // by the names devtools::kFigures gives them

std::string valueText(const devtools::Figure& figure, double value) {
    if (figure.unit == "%") {
        return std::format("{:.1f} %", value);
    }
    return figure.name == "NCCD" ? std::format("{:.2f}", value) : std::format("{:.0f}", value);
}

void sayFigure(std::string_view name, double value, std::string_view detail, bool list, std::FILE* out, MeasuredFigures& measured, const devtools::Places& places) {
    const auto figure = std::ranges::find(devtools::kFigures, name, &devtools::Figure::name);
    assert(figure != devtools::kFigures.end());
    measured[std::string{name}] = value;
    say(out, "  {}: {}{}", name, valueText(*figure, value), detail);
    const auto own = places.find(name);
    if (!list || own == places.end()) {
        return;
    }
    std::vector<std::pair<std::string_view, const devtools::Place*>> largestFirst;
    for (const auto& [key, place] : own->second) {
        largestFirst.emplace_back(key, &place);
    }
    std::ranges::stable_sort(largestFirst, std::greater{}, [](const auto& entry) { return entry.second->value; });
    for (const auto& [key, place] : largestFirst) {
        say(out, "{}", devtools::placeLine(key, *place));
    }
}

std::string functionKey(const FunctionDefinition& definition, const Scope& scope) { return std::format("{} {}", scope.relative(definition.file), definition.signature); }

void printComposition(const Findings& findings, const Scope& scope, bool list, std::FILE* out, MeasuredFigures& measured, devtools::Places& places) {
    std::set<std::string> components;
    for (const fs::path& root : scope.roots) {
        for (const auto& entry : fs::recursive_directory_iterator(root)) {
            const std::string file = entry.path().string();
            if ((entry.path().extension() == ".cpp" || entry.path().extension() == ".hpp") && scope.inSource(file)) {
                components.insert(scope.componentOf(file));
            }
        }
    }
    std::map<std::string, std::set<std::string>> dependencies;
    for (const auto& [includer, included] : findings.includes) {
        if (scope.inSource(includer) && scope.inSource(included) && scope.componentOf(includer) != scope.componentOf(included)) {
            dependencies[scope.componentOf(includer)].insert(scope.componentOf(included));
        }
    }
    const auto reachOver = [&components](const std::map<std::string, std::set<std::string>>& edges) {
        std::map<std::string, std::set<std::string>> reach;
        for (const std::string& component : components) {
            std::set<std::string>    reached{component};
            std::vector<std::string> pending{component};
            while (!pending.empty()) {
                const std::string visiting = pending.back();
                pending.pop_back();
                const auto next = edges.find(visiting);
                for (const std::string& dependency : next == edges.end() ? std::set<std::string>{} : next->second) {
                    if (reached.insert(dependency).second) {
                        pending.push_back(dependency);
                    }
                }
            }
            reach[component] = std::move(reached);
        }
        return reach;
    };
    const auto ccdOf = [](const std::map<std::string, std::set<std::string>>& reach) {
        const auto reachedCounts = reach | std::views::values | std::views::transform(&std::set<std::string>::size);
        return std::ranges::fold_left(reachedCounts, 0UZ, std::plus{});
    };
    auto              reach = reachOver(dependencies);
    const std::size_t ccd   = ccdOf(reach);
    std::size_t       edges = 0, inCycles = 0;
    for (const auto& [component, reached] : reach) {
        edges += dependencies[component].size();
        if (std::ranges::any_of(reached, [&](const std::string& other) { return other != component && reach[other].contains(component); })) {
            ++inCycles;
            places["components in a cycle"][component] = devtools::Place{.value = 1.0, .where = "reaches itself through #include"};
        }
    }
    for (const auto& [includer, includedSet] : dependencies) {
        for (const std::string& included : includedSet) {
            auto withoutEdge = dependencies;
            withoutEdge[includer].erase(included);
            const std::size_t saving                                             = ccd - ccdOf(reachOver(withoutEdge));
            places["include edges"][std::format("{} -> {}", includer, included)] = devtools::Place{.value = 1.0, .where = std::format("removing it alone lowers CCD by {}", saving)};
        }
    }
    std::vector<std::pair<std::size_t, std::string>> widest;
    for (const auto& [component, reached] : reach) {
        widest.emplace_back(reached.size() - 1, component);
    }
    std::ranges::sort(widest, std::greater{});
    say(out, "composition");
    sayFigure("components", static_cast<double>(components.size()), "", list, out, measured, places);
    sayFigure("include edges", static_cast<double>(edges), "", list, out, measured, places);
    sayFigure("components in a cycle", static_cast<double>(inCycles), "", list, out, measured, places);
    sayFigure("CCD", static_cast<double>(ccd), " (the places behind it are the include edges)", list, out, measured, places);
    sayFigure("NCCD", devtools::normalisedCcd(ccd, components.size()), "", list, out, measured, places);
    std::string widestReach;
    for (const auto& [reachedCount, component] : widest | std::views::take(3)) {
        widestReach += std::format("{}{} {}", widestReach.empty() ? "" : ", ", component, reachedCount);
    }
    say(out, "  widest reach (other components a component's #includes reach): {}", widestReach);
    std::map<std::string, std::size_t> unitsPerLibrary;
    for (const auto& [unit, libraries] : findings.externalLibraries) {
        for (const std::string& library : libraries) {
            ++unitsPerLibrary[library];
        }
    }
    std::string libraryReach;
    for (const auto& [library, units] : unitsPerLibrary) {
        libraryReach += std::format("{}{} {}", libraryReach.empty() ? "" : ", ", library, units);
    }
    say(out, "  translation units under the source roots reaching each library ({} units): {}", findings.externalLibraries.size(), libraryReach);
}

void printShape(const Findings& findings, const Scope& scope, bool list, std::FILE* out, MeasuredFigures& measured, devtools::Places& places) {
    for (const auto& [where, statements] : findings.multiStatementLocals) {
        places["live mutable locals"][where.key()] = devtools::Place{.value = 1.0, .where = std::format("line {}, written in {} statements", where.line, statements)};
    }
    int unscopedWeight = 0;
    for (const auto& [where, weightAndWhat] : findings.unscopedPairs) {
        places["unscoped pairs"][where.key()] = devtools::Place{.value = static_cast<double>(weightAndWhat.first), .where = std::format("line {}, {}", where.line, weightAndWhat.second)};
        unscopedWeight += weightAndWhat.first;
    }
    std::size_t constantParameters = 0;
    for (const auto& [calleeUsr, sites] : findings.constantArguments) {
        const auto definition = findings.definitions.find(calleeUsr);
        if (definition == findings.definitions.end() || sites.size() < 2UZ) {
            continue; // only an own function, and only one called from two places or more, can lose a parameter to its callers
        }
        const std::vector<std::string>& first = sites.begin()->second;
        for (std::size_t index = 0; index < first.size(); ++index) {
            const bool same = !first[index].empty() && std::ranges::all_of(sites, [&](const auto& site) { return index < site.second.size() && site.second[index] == first[index]; });
            if (!same) {
                continue;
            }
            const std::vector<std::string>& names                                                                                                     = findings.parameterNames.at(calleeUsr);
            const std::string               name                                                                                                      = index < names.size() && !names[index].empty() ? names[index] : std::format("#{}", index + 1);
            places["same argument everywhere"][std::format("{} {}: {}", scope.relative(definition->second.file), definition->second.signature, name)] = devtools::Place{.value = 1.0, .where = std::format("always {}, at {} calls", first[index], sites.size())};
            ++constantParameters;
        }
    }
    std::size_t unreadFields = 0;
    for (const auto& [fieldUsr, where] : findings.fieldsDeclared) {
        // the record's USR is the field's without its last component
        const std::string recordUsr = fieldUsr.substr(0, fieldUsr.rfind('@'));
        const std::string fieldName = where.what.substr(where.what.rfind(':') + 1);
        if (findings.fieldsRead.contains(fieldUsr) || findings.fieldNamesReadDependently.contains(fieldName) || findings.recordsCompared.contains(recordUsr)) {
            continue;
        }
        places["write-only fields"][where.key()] = devtools::Place{.value = 1.0, .where = std::format("line {}, never read", where.line)};
        ++unreadFields;
    }
    std::size_t flagParameters = 0;
    for (const auto& [calleeUsr, sites] : findings.constantArguments) {
        const auto definition = findings.definitions.find(calleeUsr);
        if (definition == findings.definitions.end()) {
            continue;
        }
        const std::size_t parameterCount = std::ranges::max(sites | std::views::values | std::views::transform(&std::vector<std::string>::size));
        for (std::size_t index = 0; index < parameterCount; ++index) {
            const auto literal   = [&](std::string_view value) { return std::ranges::count_if(sites | std::views::values, [&](const auto& arguments) { return index < arguments.size() && arguments[index] == value; }); };
            const auto withTrue  = literal("true");
            const auto withFalse = literal("false");
            if (withTrue + withFalse == 0) {
                continue;
            }
            const std::vector<std::string>& names                                                                                           = findings.parameterNames.at(calleeUsr);
            const std::string               name                                                                                            = index < names.size() && !names[index].empty() ? names[index] : std::format("#{}", index + 1);
            places["flag arguments"][std::format("{} {}: {}", scope.relative(definition->second.file), definition->second.signature, name)] = devtools::Place{.value = 1.0, .where = std::format("true at {}, false at {} of {} calls", withTrue, withFalse, sites.size())};
            ++flagParameters;
        }
    }
    std::size_t nestedTernaryCount = 0;
    for (const auto& [functionUsr, offsets] : findings.nestedTernaries) {
        if (const auto definition = findings.definitions.find(functionUsr); definition != findings.definitions.end()) {
            places["nested ternaries"][std::format("{} {}", scope.relative(definition->second.file), definition->second.signature)] = devtools::Place{.value = static_cast<double>(offsets.size()), .where = std::format("line {}", definition->second.firstLine)};
            nestedTernaryCount += offsets.size();
        }
    }
    say(out, "shape");
    sayFigure("live mutable locals", static_cast<double>(findings.multiStatementLocals.size()), " (locals written in more than one top-level statement)", list, out, measured, places);
    sayFigure("write-only fields", static_cast<double>(unreadFields), " (fields under the source roots that no own code reads, tests included)", list, out, measured, places);
    sayFigure("nested ternaries", static_cast<double>(nestedTernaryCount), " (a ?: holding another in the same expression)", list, out, measured, places);
    sayFigure("flag arguments", static_cast<double>(flagParameters), " (parameters some call fills with a literal true or false)", list, out, measured, places);
    sayFigure("same argument everywhere", static_cast<double>(constantParameters), " (parameters every call passes the same constant)", list, out, measured, places);
    sayFigure("unscoped pairs", static_cast<double>(unscopedWeight), std::format(" ({} pairs opened and closed by hand in one function; +1 per exit or branch that could skip the closing)", findings.unscopedPairs.size()), list, out, measured, places);
}

void printVolume(const Findings& findings, const Scope& scope, bool list, std::FILE* out, MeasuredFigures& measured, devtools::Places& places) {
    FileFigures projectFigures;
    std::size_t restatementCount    = 0;
    std::size_t repeatedWritings    = 0;
    std::size_t repeatedExpressions = 0;
    for (const auto& [path, figures] : findings.files) {
        const std::string file = scope.relative(path);
        projectFigures.tokens += figures.tokens;
        projectFigures.sloc += figures.sloc;
        projectFigures.commentLines += figures.commentLines;
        projectFigures.commentWords += figures.commentWords;
        projectFigures.over160 += figures.over160;
        projectFigures.over250 += figures.over250;
        restatementCount += figures.restatements.size();
        const std::string fileVolume           = std::format("{} tokens, {} comment words, {} SLOC", figures.tokens, figures.commentWords, figures.sloc);
        places["tokens + comment words"][file] = devtools::Place{.value = static_cast<double>(figures.tokens + figures.commentWords), .where = fileVolume};
        for (const auto& [text, times] : figures.repeatedExpressions) {
            places["repeated expressions"][std::format("{}: {}", file, text)] = devtools::Place{.value = static_cast<double>(times), .where = std::format("written {} times", times)};
            repeatedWritings += times - 1UZ;
            ++repeatedExpressions;
        }
        for (const auto& [line, text] : figures.restatements) {
            places["restating comments"][std::format("{}: {}", file, text)] = devtools::Place{.value = 1.0, .where = std::format("line {}", line)};
        }
        if (figures.over160 > 0) {
            places["lines over 160 characters"][file] = devtools::Place{.value = static_cast<double>(figures.over160), .where = "lines in this file"};
        }
        if (figures.over250 > 0) {
            places["lines over 250 characters"][file] = devtools::Place{.value = static_cast<double>(figures.over250), .where = "lines in this file"};
        }
    }
    say(out, "volume");
    const std::size_t readable = projectFigures.tokens + projectFigures.commentWords;
    const std::string parts    = std::format(" ({} tokens and {} comment words in {} files; {} comment lines)", projectFigures.tokens, projectFigures.commentWords, findings.files.size(), projectFigures.commentLines);
    sayFigure("tokens + comment words", static_cast<double>(readable), parts, list, out, measured, places);
    sayFigure("SLOC", static_cast<double>(projectFigures.sloc), " (lines holding code)", list, out, measured, places);
    sayFigure("lines over 160 characters", static_cast<double>(projectFigures.over160), "", list, out, measured, places);
    sayFigure("lines over 250 characters", static_cast<double>(projectFigures.over250), "", list, out, measured, places);
    sayFigure("repeated expressions", static_cast<double>(repeatedWritings), std::format(" ({} expressions of {} tokens or more written {} times or more in one file; counted beyond the first writing)", repeatedExpressions, kRepeatedTokens, kRepeatedTimes), list, out, measured, places);
    say(out, "comments");
    sayFigure("restating comments", static_cast<double>(restatementCount), "", list, out, measured, places);
}

void printLocality(const Findings& findings, const Scope& scope, bool list, std::FILE* out, MeasuredFigures& measured, devtools::Places& places) {
    std::size_t                                  callsWithin = 0, callsAcross = 0;
    std::map<std::string, std::set<std::string>> sourceCallers; // callee -> callers under the source roots
    std::map<std::string, std::size_t>           sourceCallSites;
    std::set<std::string>                        testedCallees;
    for (const Call& call : findings.calls) {
        const auto callee = findings.definitions.find(call.calleeUsr);
        if (callee == findings.definitions.end() || call.callerUsr == call.calleeUsr) {
            continue;
        }
        if (scope.inTests(call.callerFile)) {
            testedCallees.insert(call.calleeUsr);
            continue;
        }
        if (scope.componentOf(call.callerFile) == scope.componentOf(callee->second.file)) {
            ++callsWithin;
        } else {
            ++callsAcross;
            devtools::Place& pair = places["calls across components"][std::format("{} -> {}", scope.componentOf(call.callerFile), scope.componentOf(callee->second.file))];
            pair.value += 1.0;
            pair.where = "calls from the first component into the second";
        }
        sourceCallers[call.calleeUsr].insert(call.callerUsr);
        ++sourceCallSites[call.calleeUsr];
    }
    std::size_t loneCalleeCount = 0;
    for (const auto& [usr, callers] : sourceCallers) {
        const FunctionDefinition& callee  = findings.definitions.at(usr);
        const auto                caller  = findings.definitions.find(*callers.begin());
        const bool                crosses = caller != findings.definitions.end() && scope.componentOf(caller->second.file) != scope.componentOf(callee.file);
        if (callers.size() == 1 && sourceCallSites[usr] == 1 && crosses && !testedCallees.contains(usr)) {
            ++loneCalleeCount;
            const std::string key       = std::format("{} <- {}", functionKey(callee, scope), scope.componentOf(caller->second.file));
            places["lone callees"][key] = devtools::Place{.value = 1.0, .where = std::format("line {}, {} lines", callee.firstLine, callee.lastLine - callee.firstLine + 1)};
        }
    }
    std::size_t unusedCount = 0;
    for (const auto& [usr, definition] : findings.definitions) {
        if (!definition.reachedOtherwise && !findings.functionsReferenced.contains(usr) && !findings.namesUnresolved.contains(definition.name)) {
            ++unusedCount;
            places["unused functions"][functionKey(definition, scope)] = devtools::Place{.value = 1.0, .where = std::format("line {}, {} lines", definition.firstLine, definition.lastLine - definition.firstLine + 1)};
        }
    }
    const std::size_t callCount = std::max<std::size_t>(callsWithin + callsAcross, 1);
    say(out, "locality");
    const double acrossShare = 100.0 * static_cast<double>(callsAcross) / static_cast<double>(callCount);
    sayFigure("calls across components", acrossShare, std::format(" ({} of {} calls to own functions)", callsAcross, callsWithin + callsAcross), list, out, measured, places);
    sayFigure("lone callees", static_cast<double>(loneCalleeCount), " (one call site, in another component, no test)", list, out, measured, places);
    sayFigure("unused functions", static_cast<double>(unusedCount), " (defined under the source roots, named by no own code, tests included)", list, out, measured, places);
}

struct CognitiveScore {
    std::size_t complexity = 0;
    std::size_t penalised  = 0; // increments that carry a nesting penalty: each grows by the call depth when inlined
};

// lines "<complexity> <penalised> <file>:<line> <name>" from devtools/code-metrics.sh --cognitive, keyed by "<file>:<line>"
std::map<std::string, std::pair<CognitiveScore, std::string>> cognitiveScoresIn(const std::string& scoresPath) {
    std::map<std::string, std::pair<CognitiveScore, std::string>> scores;
    std::ifstream                                                 scoresFile{scoresPath};
    for (std::string line; std::getline(scoresFile, line);) {
        std::istringstream fields{line};
        CognitiveScore     score;
        std::string        location;
        std::string        name;
        if (!(fields >> score.complexity >> score.penalised >> location >> name)) {
            continue;
        }
        const std::size_t lineSeparator = location.rfind(':');
        const std::string key           = std::format("{}:{}", fs::weakly_canonical(location.substr(0, lineSeparator)).string(), location.substr(lineSeparator + 1));
        scores.try_emplace(key, score, name);
    }
    return scores;
}

void printComplexity(const Findings& findings, const Scope& scope, const std::string& scoresPath, bool list, std::FILE* out, MeasuredFigures& measured, devtools::Places& places) {
    const auto                            scores = cognitiveScoresIn(scoresPath);
    std::map<std::string, CognitiveScore> scoreByUsr;
    std::set<std::string>                 matchedKeys;
    for (const auto& [usr, definition] : findings.definitions) {
        const std::string key    = std::format("{}:{}", definition.file, definition.firstLine);
        const auto        scored = scores.find(key);
        if (scored != scores.end()) {
            matchedKeys.insert(key);
            const auto plain                                               = static_cast<double>(scored->second.first.complexity);
            places["cognitive complexity"][functionKey(definition, scope)] = devtools::Place{.value = plain, .where = std::format("line {}", definition.firstLine)};
        }
        if (scope.inSource(definition.file)) { // clang-tidy reports no function that scores 0
            scoreByUsr.emplace(usr, scored == scores.end() ? CognitiveScore{} : scored->second.first);
        }
    }
    struct FoldedCall {
        std::string calleeUsr;
        unsigned    nestingDepth;
    };
    std::map<std::string, std::vector<std::pair<std::string, unsigned>>> sourceSites; // callee -> (caller, depth) per call site under the source roots
    std::set<std::string>                                                testedCallees;
    for (const Call& call : findings.calls) {
        if (scope.inTests(call.callerFile)) {
            testedCallees.insert(call.calleeUsr);
        } else if (scope.inSource(call.callerFile)) {
            sourceSites[call.calleeUsr].emplace_back(call.callerUsr, call.nestingDepth);
        }
    }
    std::map<std::string, std::vector<FoldedCall>> foldsByCaller;
    for (const auto& [calleeUsr, sites] : sourceSites) {
        const bool isSingleUse = sites.size() == 1UZ && !testedCallees.contains(calleeUsr) && sites.front().first != calleeUsr;
        if (isSingleUse && scoreByUsr.contains(calleeUsr) && scoreByUsr.contains(sites.front().first)) {
            foldsByCaller[sites.front().first].push_back(FoldedCall{.calleeUsr = calleeUsr, .nestingDepth = sites.front().second});
        }
    }

    std::map<std::string, CognitiveScore> splitInvariant;
    std::set<std::string>                 onCallPath;
    std::set<std::string>                 foldedAway;
    const auto                            foldedScoreOf = [&](this const auto& foldedScore, const std::string& usr) -> CognitiveScore {
        if (const auto known = splitInvariant.find(usr); known != splitInvariant.end()) {
            return known->second;
        }
        onCallPath.insert(usr);
        CognitiveScore total = scoreByUsr.at(usr);
        for (const FoldedCall& folded : foldsByCaller[usr]) {
            if (onCallPath.contains(folded.calleeUsr)) {
                continue;
            }
            const CognitiveScore inner = foldedScore(folded.calleeUsr);
            total.complexity += inner.complexity + folded.nestingDepth * inner.penalised;
            total.penalised += inner.penalised;
            foldedAway.insert(folded.calleeUsr);
        }
        onCallPath.erase(usr);
        return splitInvariant[usr] = total;
    };
    for (const auto& [usr, score] : scoreByUsr) {
        std::ignore = foldedScoreOf(usr);
    }

    std::size_t plainSum = 0;
    for (const auto& [key, scoreAndName] : scores) {
        plainSum += scoreAndName.first.complexity;
    }
    std::size_t                                                    splitInvariantSum = plainSum;
    std::vector<std::tuple<std::size_t, std::size_t, std::string>> roots; // split-invariant, plain, name
    for (const auto& [usr, score] : scoreByUsr) {
        splitInvariantSum -= score.complexity;
        if (foldedAway.contains(usr)) {
            continue;
        }
        splitInvariantSum += splitInvariant.at(usr).complexity;
        if (splitInvariant.at(usr).complexity == 0) {
            continue;
        }
        const FunctionDefinition& definition = findings.definitions.at(usr);
        roots.emplace_back(splitInvariant.at(usr).complexity, score.complexity, definition.name);
        const auto folded                                                    = static_cast<double>(splitInvariant.at(usr).complexity);
        places["split-invariant complexity"][functionKey(definition, scope)] = devtools::Place{.value = folded, .where = std::format("line {}, {} before folding", definition.firstLine, score.complexity)};
    }
    std::ranges::sort(roots, std::greater{});
    std::string highest;
    for (const auto& [complexity, plain, where] : roots | std::views::take(5)) {
        highest += std::format("{}{} {}", highest.empty() ? "" : ", ", where, complexity);
    }
    say(out, "complexity (sources; code-metrics.sh --cognitive)");
    std::vector<std::pair<std::size_t, std::string>> plainScores;
    for (const auto& [key, scoreAndName] : scores) {
        plainScores.emplace_back(scoreAndName.first.complexity, scoreAndName.second);
    }
    std::ranges::sort(plainScores, std::greater{});
    std::string highestPlain;
    for (const auto& [complexity, name] : plainScores | std::views::take(5)) {
        highestPlain += std::format("{}{} {}", highestPlain.empty() ? "" : ", ", name, complexity);
    }
    sayFigure("cognitive complexity", static_cast<double>(plainSum), std::format(" ({} functions scored)", scores.size()), list, out, measured, places);
    say(out, "  highest: {}", highestPlain);
    const std::string foldedCount = std::format(" ({} single-use callees folded in at their call depth)", foldedAway.size());
    sayFigure("split-invariant complexity", static_cast<double>(splitInvariantSum), foldedCount, list, out, measured, places);
    say(out, "  highest split-invariant: {}", highest);
    say(out, "  scored functions without a definition found: {}", scores.size() - matchedKeys.size());
    for (const auto& [key, scoreAndName] : list ? scores : decltype(scores){}) {
        if (!matchedKeys.contains(key)) {
            say(out, "    without a definition: {} {}", key, scoreAndName.second);
        }
    }
}

void printAmbientReach(const Findings& findings, const Scope& scope, bool list, std::FILE* out, MeasuredFigures& measured, devtools::Places& places) {
    std::map<std::string, std::set<std::string>> reach; // function USR -> categories, direct and through own callees
    for (const auto& [usr, sources] : findings.ambientSources) {
        for (const auto& [category, names] : sources) {
            reach[usr].insert(category);
        }
    }
    std::map<std::string, std::set<std::string>> callersOf;
    for (const Call& call : findings.calls) {
        if (scope.inSource(call.callerFile) && findings.definitions.contains(call.callerUsr) && findings.definitions.contains(call.calleeUsr)) {
            callersOf[call.calleeUsr].insert(call.callerUsr);
        }
    }
    std::vector<std::string> pending;
    for (const auto& [usr, categories] : reach) {
        pending.push_back(usr);
    }
    while (!pending.empty()) {
        const std::string callee = pending.back();
        pending.pop_back();
        for (const std::string& caller : callersOf[callee]) {
            const std::size_t before = reach[caller].size();
            reach[caller].insert(reach[callee].begin(), reach[callee].end());
            if (reach[caller].size() != before) {
                pending.push_back(caller);
            }
        }
    }

    std::size_t                        sourceFunctions = 0;
    std::size_t                        honestFunctions = 0;
    std::size_t                        coreReaching    = 0;
    std::map<std::string, std::size_t> functionsPerCategory;
    for (const auto& [usr, definition] : findings.definitions) {
        if (!scope.inSource(definition.file)) {
            continue;
        }
        ++sourceFunctions;
        const auto reached = reach.find(usr);
        if (reached == reach.end() || reached->second.empty()) {
            ++honestFunctions;
            continue;
        }
        const bool inCore = scope.inCore(definition.file);
        coreReaching += inCore ? 1 : 0;
        std::string categories;
        for (const std::string& category : reached->second) {
            ++functionsPerCategory[category];
            categories += std::format("{}{}", categories.empty() ? "" : ", ", category);
        }
        std::string directSources;
        if (const auto direct = findings.ambientSources.find(usr); direct != findings.ambientSources.end()) {
            for (const auto& [category, names] : direct->second) {
                for (const std::string& name : names) {
                    directSources += std::format("{}{}", directSources.empty() ? "" : ", ", name);
                }
            }
        }
        const std::string     direct = directSources.empty() ? std::string{} : std::format(" (directly: {})", directSources);
        const devtools::Place reaching{.value = 1.0, .where = std::format("line {}: {}{}", definition.firstLine, categories, direct)};
        places["functions without ambient reach"][functionKey(definition, scope)] = reaching;
        if (inCore) {
            places["core functions with ambient reach"][functionKey(definition, scope)] = reaching;
        }
    }
    std::string perCategory;
    for (const auto& [category, count] : functionsPerCategory) {
        perCategory += std::format("{}{} {}", perCategory.empty() ? "" : ", ", category, count);
    }
    say(out, "ambient reach (sources; write-only logging excluded; indirect calls not resolved, so a lower bound)");
    const double      honestShare = 100.0 * static_cast<double>(honestFunctions) / static_cast<double>(std::max<std::size_t>(sourceFunctions, 1));
    const std::string honestCount = std::format(" ({} of {}; the places are the {} that reach something)", honestFunctions, sourceFunctions, sourceFunctions - honestFunctions);
    sayFigure("functions without ambient reach", honestShare, honestCount, list, out, measured, places);
    say(out, "  functions reaching each: {}", perCategory);
    sayFigure("core functions with ambient reach", static_cast<double>(coreReaching), "", list, out, measured, places);
}

constexpr std::string_view kNotMeasured = "not measured (code-metrics.sh --cognitive)";

void printSummary(const MeasuredFigures& measured, const devtools::UnitCounts& units) {
    std::println("translation units parsed: {} of {}", units.wanted - units.unparsable, units.wanted);
    for (const devtools::Group& group : devtools::kGroups) {
        if (!group.measuredBy.empty()) {
            std::println("({}) {}: {}", group.number, group.title, group.measuredBy);
            continue;
        }
        std::println("({}) {}", group.number, group.title);
        for (const devtools::Figure& figure : devtools::kFigures) {
            if (figure.group != group.number) {
                continue;
            }
            const auto value = measured.find(figure.name);
            std::println("  {}: {}", figure.name, value == measured.end() ? std::string{kNotMeasured} : valueText(figure, value->second));
            std::println("      {}", figure.meaning);
        }
    }
}

void printTable(const MeasuredFigures& measured) {
    std::println("| | figure | value | better |");
    std::println("| --- | --- | ---: | --- |");
    for (const devtools::Figure& figure : devtools::kFigures) {
        const auto value = measured.find(figure.name);
        std::println("| ({}) | {} | {} | {} |", figure.group, figure.name, value == measured.end() ? std::string{"not measured"} : valueText(figure, value->second), devtools::betterName(figure.better));
    }
}

/// each figure before and now; in text, under a figure that moved, the places that moved it (top 5, all with --list)
/// and, when it moved the worse way, the usual remedy; the Markdown table stays factual for a commit message
void printComparison(const MeasuredFigures& measured, const std::map<std::string, double>& earlier, const devtools::Places& places, const devtools::Places& earlierPlaces, bool markdown, bool list) {
    constexpr std::size_t kPlacesShown = 5;
    if (markdown) {
        std::println("| | figure | before | after | change | |");
        std::println("| --- | --- | ---: | ---: | ---: | --- |");
    }
    std::set<std::string_view> placesShown;
    for (const devtools::Figure& figure : devtools::kFigures) {
        const auto now  = measured.find(figure.name);
        const auto then = earlier.find(std::string{figure.name});
        // a report holds a figure at its printed precision, so the value now is compared at that precision too
        const devtools::Measured after      = now == measured.end() ? devtools::Measured{} : devtools::figuresIn(std::format("  {}: {}", figure.name, valueText(figure, now->second))).begin()->second;
        const devtools::Measured before     = then == earlier.end() ? devtools::Measured{} : devtools::Measured{then->second};
        const devtools::Verdict  verdict    = devtools::verdictOf(figure.better, before, after);
        const std::string        beforeText = before ? valueText(figure, *before) : std::string{"—"};
        const std::string        afterText  = after ? valueText(figure, *after) : std::string{"not measured"};
        std::string              change;
        if (before && after && verdict != devtools::Verdict::unchanged) {
            const std::optional<double> percent = devtools::percentChange(*before, *after);
            change                              = percent ? std::format("{:+.1f} %", *percent) : std::format("{:+.0f}", *after - *before);
        }
        if (markdown) {
            std::println("| ({}) | {} | {} | {} | {} | {} |", figure.group, figure.name, beforeText, afterText, change, devtools::verdictName(verdict));
            continue;
        }
        std::println("({}) {:<40} {:>12} -> {:<12} {:>9}  {}", figure.group, figure.name, beforeText, afterText, change, devtools::verdictName(verdict));
        const bool moved = verdict == devtools::Verdict::better || verdict == devtools::Verdict::worse || verdict == devtools::Verdict::changed;
        if (!moved || figure.placesOf.empty()) {
            continue;
        }
        if (earlierPlaces.empty()) {
            std::println("      where: not known, the earlier report names no places (keep a --list report to see them)");
        } else if (!placesShown.insert(figure.placesOf).second) {
            std::println("      places: as under {}", figure.placesOf);
        } else {
            const auto placesNow                                  = places.find(figure.placesOf);
            const auto placesBefore                               = earlierPlaces.find(figure.placesOf);
            using PlaceMap                                        = std::map<std::string, devtools::Place, std::less<>>;
            const PlaceMap&                          placesThen   = placesBefore == earlierPlaces.end() ? PlaceMap{} : placesBefore->second;
            const PlaceMap&                          placesNowMap = placesNow == places.end() ? PlaceMap{} : placesNow->second;
            const std::vector<devtools::PlaceChange> changes      = devtools::placeChanges(placesThen, placesNowMap);
            for (const devtools::PlaceChange& placeChange : changes | std::views::take(list ? changes.size() : kPlacesShown)) {
                const auto& [key, valueBefore, valueAfter, where] = placeChange;
                const bool             counted                    = valueBefore.value_or(1.0) != 1.0 || valueAfter.value_or(1.0) != 1.0; // a listed item counts 1
                const std::string_view kind                       = [&] -> std::string_view {
                    if (!valueBefore) {
                        return "new ";
                    }
                    if (!valueAfter) {
                        return "gone";
                    }
                    return *valueAfter > *valueBefore ? "rose" : "fell";
                }();
                const std::string values = [&] -> std::string {
                    if (!counted) {
                        return {};
                    }
                    if (valueBefore && valueAfter) {
                        return std::format(": {:.0f} -> {:.0f}", *valueBefore, *valueAfter);
                    }
                    return std::format(": {:.0f}", valueAfter.value_or(valueBefore.value_or(0.0)));
                }();
                std::println("      {}  {}{}   ({}{})", kind, key, values, valueAfter ? "" : "was ", where);
            }
            if (!list && changes.size() > kPlacesShown) {
                std::println("      … and {} more (all with --list)", changes.size() - kPlacesShown);
            }
        }
        if (verdict == devtools::Verdict::worse && !figure.remedy.empty()) {
            std::println("      {}", figure.remedy);
        }
    }
}

} // namespace

int main(int argc, char** argv) {
    const std::vector<std::string_view> arguments(argv + 1, argv + argc);
    const auto                          has = [&arguments](std::string_view flag) { return std::ranges::contains(arguments, flag); };
    if (has("--help") || has("-h")) {
        std::print("{}", kMeasurementMethod);
        return 0;
    }
    if (has("--figures")) {
        for (const devtools::Group& group : devtools::kGroups) {
            std::println("({}) {}{}", group.number, group.title, group.measuredBy.empty() ? std::string{} : std::format(": {}", group.measuredBy));
            for (const devtools::Figure& figure : devtools::kFigures) {
                if (figure.group == group.number) {
                    std::println("  {}{}  (better: {})", figure.name, figure.unit.empty() ? std::string{} : std::format(" [{}]", figure.unit), devtools::betterName(figure.better));
                    std::println("      {}", figure.meaning);
                }
            }
        }
        return 0;
    }
    const std::string_view                       jobsText = devtools::valueAfter(arguments, "--jobs", "4");
    const std::string                            comparePath{devtools::valueAfter(arguments, "--compare", "")};
    std::size_t                                  jobs = 0;
    std::optional<std::map<std::string, double>> earlier;
    devtools::Places                             earlierPlaces;
    if (!comparePath.empty()) {
        std::ifstream     reportFile{comparePath};
        std::stringstream report;
        report << reportFile.rdbuf();
        if (reportFile) {
            earlier       = devtools::figuresIn(report.str());
            earlierPlaces = devtools::placesIn(report.str());
        }
    }
    Scope scope{.repository = fs::current_path(), .roots = {}, .excluded = {}, .core = {}};
    for (std::size_t i = 0; i + 1 < arguments.size(); ++i) {
        const fs::path value = fs::weakly_canonical(std::string{arguments[i + 1]});
        if (arguments[i] == "--root") {
            scope.roots.push_back(value);
        } else if (arguments[i] == "--exclude") {
            scope.excluded.push_back(value);
        } else if (arguments[i] == "--core") {
            scope.core = value;
        }
    }
    if (arguments.empty() || arguments.front().starts_with("-") || scope.roots.empty() || std::from_chars(jobsText.data(), jobsText.data() + jobsText.size(), jobs).ec != std::errc{} || jobs == 0U || (!comparePath.empty() && !earlier)) {
        std::println(stderr, "usage: code-metrics <clang-build-dir> --root <dir>... [--exclude <dir>]... [--core <dir>] [--jobs <n>] [--list] [--cognitive-scores <file>] [--summary | --table | --compare <report> [--table] [--list]]; --help explains the figures");
        return 2;
    }
    const bool list    = has("--list");
    const bool summary = has("--summary");
    const bool table   = has("--table");

    Findings              allFindings;
    std::mutex            claimLock;
    std::set<std::string> claimedFiles;
    const auto            unitCount = devtools::forEachUnit(
        arguments.front(), [&scope](const std::string& file) { return scope.own(file); }, jobs, CXTranslationUnit_KeepGoing, [&](CXTranslationUnit unit, CXCompileCommand command) { return analyse(unit, command, scope, claimLock, claimedFiles); },
        [&allFindings](Findings unitFindings) {
            allFindings.definitions.merge(unitFindings.definitions);
            allFindings.calls.merge(unitFindings.calls);
            allFindings.includes.merge(unitFindings.includes);
            allFindings.files.merge(unitFindings.files);
            allFindings.multiStatementLocals.merge(unitFindings.multiStatementLocals);
            allFindings.unscopedPairs.merge(unitFindings.unscopedPairs);
            for (auto& [callee, sites] : unitFindings.constantArguments) {
                allFindings.constantArguments[callee].merge(sites);
            }
            allFindings.parameterNames.merge(unitFindings.parameterNames);
            allFindings.fieldsDeclared.merge(unitFindings.fieldsDeclared);
            allFindings.fieldsRead.merge(unitFindings.fieldsRead);
            allFindings.fieldNamesReadDependently.merge(unitFindings.fieldNamesReadDependently);
            allFindings.recordsCompared.merge(unitFindings.recordsCompared);
            allFindings.functionsReferenced.merge(unitFindings.functionsReferenced);
            allFindings.namesUnresolved.merge(unitFindings.namesUnresolved);
            for (auto& [function, offsets] : unitFindings.nestedTernaries) {
                allFindings.nestedTernaries[function].merge(offsets);
            }
            allFindings.externalLibraries.merge(unitFindings.externalLibraries);
            allFindings.ambientSources.merge(unitFindings.ambientSources);
            allFindings.unitsWithParseErrors.merge(unitFindings.unitsWithParseErrors);
        });
    if (!unitCount) {
        std::println(stderr, "code-metrics: no compile_commands.json in {}", arguments.front());
        return 2;
    }

    std::FILE* const out = summary || table || earlier ? nullptr : stdout;
    MeasuredFigures  measured;
    devtools::Places places;
    say(out, "translation units parsed: {} of {}", unitCount->wanted - unitCount->unparsable, unitCount->wanted);
    say(out, "translation units with parse errors (their figures are incomplete): {}", allFindings.unitsWithParseErrors.size());
    for (const std::string& unitSource : list ? allFindings.unitsWithParseErrors : std::set<std::string>{}) {
        say(out, "  {}", scope.relative(unitSource));
    }
    say(out, "(1) {}", devtools::kGroups[0].title);
    printComposition(allFindings, scope, list, out, measured, places);
    printLocality(allFindings, scope, list, out, measured, places);
    printAmbientReach(allFindings, scope, list, out, measured, places);
    say(out, "(2) {}", devtools::kGroups[1].title);
    if (const std::string_view scoresPath = devtools::valueAfter(arguments, "--cognitive-scores", ""); !scoresPath.empty()) {
        printComplexity(allFindings, scope, std::string{scoresPath}, list, out, measured, places);
    } else {
        say(out, "complexity: {}", kNotMeasured);
    }
    printShape(allFindings, scope, list, out, measured, places);
    say(out, "(3) {}", devtools::kGroups[2].title);
    printVolume(allFindings, scope, list, out, measured, places);
    for (const devtools::Group& group : devtools::kGroups | std::views::drop(3)) {
        say(out, "({}) {}: {}", group.number, group.title, group.measuredBy);
    }

    if (earlier) {
        printComparison(measured, *earlier, places, earlierPlaces, table, list);
    } else if (summary) {
        printSummary(measured, *unitCount);
    } else if (table) {
        printTable(measured);
    }
}
