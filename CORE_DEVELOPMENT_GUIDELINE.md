# Core Development Guidelines

### Preamble

Conway's law (Conway, 1968; Muratori, 2022) says that the structure of software reflects the communication patterns of the people who create it,
influencing both the system's architecture and its progression. Just as a garden needs constant attention to thrive, our software
and community require ongoing care. Our goal is to foster a sustainable environment where GNU Radio and its creators can flourish,
ensuring our systems are not only functional and efficient but also inviting and maintainable over the long term. Our core goals are:

- **Core-Goal №1: Software must be fit-for-purpose and remain easily adaptable to changing requirements.**
  - specifically for GNU Radio: we want to write Software-Defined-Radio (SDR) applications that can be easily adapted to new functions (e.g. protocols, schemes, ...) on the same hardware.
- **Core-Goal №2: be a friendly, empathetic and helpful person in writing software for other users**
- **Core-Goal №3: write lean software whose cost to read, change, run and debug is measured, not assumed**

'Lean' comes from the [Toyota Production System](https://global.toyota/en/company/vision-and-philosophy/production-system/):
remove waste ([muda](<https://en.wikipedia.org/wiki/Muda_(Japanese_term)>)), inconsistency ([mura](<https://en.wikipedia.org/wiki/Mura_(Japanese_term)>)) and overburden ([muri](<https://en.wikipedia.org/wiki/Muri_(Japanese_term)>)).
For code, that means quantifiable metrics rather than 'best practice' incantations.
This document gives the reasons and the human-readable detail;
[`CLAUDE.md`](CLAUDE.md) holds the same conventions as a checkable rule set (for people and for LLM coding agents), and the CI quality checks report the figures below on every pull request.

### Target Audience

We distinguish four application classes, which help determine the amount of rigour necessary in coding practice:

- **Application Class 0**: _For those tinkering with GR and writing private user code:_
  - Choose your style, conventions, rules, and guidelines that fit you best.
  - When seeking help, for better support and feedback:
    check tutorials and examples for existing solutions;
    be concise, avoid unrelated topics, and provide a minimum viable example ([MVP](https://en.wikipedia.org/wiki/Minimum_viable_product));
    adhere to the [Code of Conduct](CODE_OF_CONDUCT.md) and be courteous – most GNU Radio helpers are volunteers;
    use the [Core Naming Guidelines](CORE_NAMING_GUIDELINE.md) for easier readability.
  - Optionally, learn modern C++: [CppCon](https://cppcon.org/)'s ['Back to Basics' track](https://www.youtube.com/watch?v=Bt3zcJZIalk&list=PLHTh1InhhwT4TJaHBVWzvBOYhp27UO7mI&pp=iAQB) and [slides](https://github.com/CppCon/CppCon2023), and [Casey Muratori's channel](https://www.youtube.com/@MollyRocket).
- **Application Class 1**: _Developing Out-Of-Tree (OOT) Modules, Blocks, or Schedulers for others:_
  - As above, plus the [C++ Core Guidelines](https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines) for safe syntax in modern C++, and optionally the design section below.
- **Application Class 2**: _Developing for GNU Radio Core BlockLib with long-term maintenance support interest (more than 2-3 years):_
  - Adhere to the [Code of Conduct](CODE_OF_CONDUCT.md), [Core Naming Guidelines](CORE_NAMING_GUIDELINE.md), [Contribution Guidelines](CONTRIBUTING.md) and [C++ Core Guidelines](https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines).
  - Before developing a new block, consult the core maintainers via the [issue tracker](./issues) and check
    that the feature does not exist already, or cannot be composed from existing blocks (GR 4.0's compile-time merging offers this for free),
    which existing block is the closest template, and where the new functionality belongs in the BlockLib.
  - Ensure that your code **passes all CI tests** (e.g. through a draft pull request), generates no compiler warning, and is formatted (`./formatFiles.sh`).
- **Application Class 3**: _Developing new features in the GNU Radio Core:_
  - This document is for you: maintainers and contributors to the core BlockLib, new schedulers, or new core features.
    These guidelines are more formalised to promote sustainable coding practice and long-term maintainability, and require some experience with modern C++.

## Purposeful and Performance Metric Prioritising Coding

Striking a balance between ideal code and practical functionality is key. 'W.A.R.M.E.D' is a memorisation aid, after
Casey Muratori's six costs of code (Muratori, c. 2021), that turns this into actionable steps: producing robust, functional and efficient
code without succumbing to over-engineering or speculative features. Each step names the _metric_ that measures it:

1. **Write** code that directly addresses tangible, real-world problems. Embrace simplicity and avoid adding complexity that
   isn't necessary. Keep tasks small enough to be handed over, so others can continue the work if needed.
   _metric: effort per task_
2. **Agree** to well-established, field-tested designs with empirical backing, discarding unsupported unquantifiable ideas.
   Focus discussions on substantive challenges rather than trivial details ([bike-shedding](https://en.wikipedia.org/wiki/Law_of_triviality)).
   Instead of prolonged design speculation, quick, benchmarked proofs of concept are preferred, acknowledging that only one, or perhaps none, may succeed.
   _metric: time spent on design discussions_
3. **Readable** code should be well-written prose, with each line and function mirroring the clarity and conciseness
   of a sentence in a technical publication. Reduce the reading volume — tokens plus comment words — rather than physical lines (see "Measuring it" below).
   _metric: reading volume, cognitive complexity_
4. **Modify** and add new features or expand APIs only when there is a clear and present need. Refrain from incorporating
   enhancements for hypothetical scenarios; "add it later" should a component become essential. This not only avoids unnecessary bloat
   but the simplicity of "adding it later" is also a litmus test to assess whether the software is composable and lean.
   _metric: files and components a change touches_
5. **Execute with Efficiency**: prioritise performance and reliability from the outset. Optimisation must be grounded in actual micro-
   and macro benchmarks and validated through complementary functional (unit-)testing.
   _metric (application specific): e.g. instructions, throughput, latency, jitter, CPU usage, power consumption_
6. **Debug Strategically & use appropriate Tools**: tackle uncommon corner-cases and outliers at the end. Static code analysis tools (linters),
   AI-based generative tools, and compilers are typically more proficient than humans at optimising problems on a small to medium scale.
   Thus shift your attention from micro- to addressing broader, macro-level optimisations, complex algorithms, and architectural designs
   that are beyond the capability of those tools. _metric: e.g. test coverage of the changed lines_

Each step comes with inherent costs: time for development, time for arguments, time to read and modify, time for execution,
time to debug extraneous API and rare corner cases. Avoid the temptation of feature creep and premature optimisation (Knuth, 1974: optimise the measured critical few per cent, not everything).
Instead, weigh the 80/20 cost-benefit, focus on areas with the most value, and engage in continuous improvement.

## Design: locality-first, compression-oriented

Complexity is a property of the whole project, not of a function (Ousterhout, 2018; Brooks, 1986: essential against accidental complexity).
A code base is good when a person can hold its theory in their head (Naur, 1985), find the code behind any behaviour in one jump,
and fix a fault without a tool explaining the code to them.
Small functions, layers and abstractions each look simple in isolation and together make the system harder:
every one is a name to learn, a file to open and a jump to follow.
Local tidiness bought with global complexity is a loss.

The figures that `devtools/code-metrics.sh` and `devtools/naming-check` report, and the CI bots post on every pull request, serve this stance.
They are indicators for the developer who owns the code, not targets:
a measure that becomes a target ceases to be a good measure (Strathern, 1997, after Goodhart).

### Compression-oriented programming

Casey Muratori's "Semantic Compression" (2014) is the method this project follows:

1. **Write the code straight first.**
   Solve the problem in place, as one sequence, in the function that needs it.
2. **Compress only what actually recurs.**
   When the same thing is written a second time, pull out what the two have in common — and only that.
   A function, type or parameter exists because two real uses asked for it, never because a future one might.
3. **The result reads like the problem.**
   What remains is the shape of the domain: the recurring pieces have names, the one-off sequences stay where they run.

Compression is the opposite of designing abstractions up front.
An abstraction invented before its second use is a guess, and a wrong guess is more expensive than the duplication it was meant to avoid (Metz, 2016).
It also explains why a recurring pattern beats a tailored one even when heavier:
the pattern is already paid for — learnt, named, tested — and every new use makes it cheaper;
a bespoke solution is paid for again by every reader.

Muratori's wider view — that the hardware and the data are the ground truth, and that habits which hide them (small polymorphic units) cost both performance and understanding — is argued in "'Clean' Code, Horrible Performance" (2023).
Its performance claims are debated (critics see a micro-benchmark generalised too far);
this project adopts the uncontested part:
code shaped by the data it handles and the sequence it runs reads better and runs faster than code shaped by an abstraction catalogue.
Performance is not traded for structure without a measurement.

| work                                                 | central idea                                                                                                                  | what GR4 takes                                                            |
| ---------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| "Designing and Evaluating Reusable Components", 2004 | an API is judged on granularity, redundancy, coupling, retention and flow control                                             | API review vocabulary; less coupling and less retention are always better |
| "Complexity and Granularity", 2014                   | "never supply a higher-level function that can't be trivially replaced by a few lower-level functions that do the same thing" | a convenience call decomposes into public lower-level calls               |
| "The Thirty-Million-Line Problem", 2015              | the stack a program stands on has grown until it makes software fragile                                                       | every dependency and layer is owned code                                  |
| "Where Does Bad Code Come From?", c. 2021            | judge code by the effort to write, agree on, read, modify, execute and debug it, not by a style catalogue                     | W.A.R.M.E.D. above; measure the cost                                      |
| "The Only Unbreakable Law", 2022                     | Conway's law (below)                                                                                                          | boundaries are costs                                                      |
| written debate with R. C. Martin, 2023               | "operation first, never operand first"                                                                                        | `enum`/`std::variant` by default, polymorphism at plugin boundaries       |
| "The Big OOPs", 2025                                 | boundaries belong around systems that see the data they need, not around domain nouns                                         | state grouped by the system that acts on it                               |

### Conway's law

Conway (1968): an organisation designs systems that copy its own communication structure.
Muratori's talk (2022) argues the converse that matters for a code base:
every boundary — team, library, module, container, service — is a communication barrier,
and splitting the problem before it is understood removes most of the design space before anyone designs.
Modularisation is a cost forced by human communication limits, not a good in itself.
The critique is real and bounds the rule:
a "law" that does not say when to split is not actionable;
libraries and containers also buy reliability and reproducibility;
and subdividing is the only way to search a solution space too large for one mind.
The project takes the part both sides accept: boundaries are a price, paid deliberately and late, where the design is understood.

### Principles: the why behind `CLAUDE.md` §2

The rules themselves are stated once, in `CLAUDE.md` §2; here is why each holds, where it comes from, and where it stops.

1. **Complexity is a project total.**
   A refactor that lowers each function's complexity by adding helpers, wrappers or files has raised the total (Ousterhout, 2018: deep modules over shallow ones).
2. **Boundaries follow understanding** (Conway; Muratori, 2022).
   A boundary that most changes must cross is in the wrong place.
   Bound: GR4's modules (`core`, `meta`, `algorithm`, `blocks/*`) and the plugin boundary exist for good reasons.
3. **Systems, not nouns; operation first** (Muratori, 2023, 2025).
   State is grouped by the system that acts on it (scheduler, graph, ports and buffers, settings);
   a closed set of kinds is an `enum class`, a table or a `std::variant`, and a class hierarchy is reserved for a plugin boundary where types are added from outside.
   Bound: where kinds share no behaviour, data abstraction suffices, which is close to this rule.
4. **Recurrence over bespoke** (Muratori, 2014; Metz, 2016).
   Recurrence must be _semantic_ (the same concern), not merely syntactic (the same look); a recurring pattern found wrong is fixed everywhere in one change.
5. **One sequence, one function; scope inside, not around** (Carmack, 2007).
   Size alone never splits a function.
   The main risk of a long function is state written from several places in one body ("complecting", Hickey, 2011);
   `const` by default, block scope and named local lambdas are the guard.
6. **Locality of behaviour** (Gross, 2020).
   The behaviour behind a call site is reachable in one jump; few global types, and every other feature's files removable with only their call sites failing (Lakos, 1996: levelised, acyclic physical design).
7. **Continuous granularity; less retention** (Muratori, 2004, 2014).
   A convenience function is a few lower-level calls a user could write themselves;
   recompute from the source of truth rather than keep a cached copy in sync.
8. **Every dependency is owned code** (Muratori, 2015): its compile time, include reach and failure modes are paid here.
9. **Hiding is deliberate, not a default.**
   Complete hiding of implementation is a tool for real boundaries (a library others build on); inside the project, plain data with public fields.
10. **Troubleshootable without an LLM** (Naur, 1985 — the program is the theory its people hold):
    a backtrace names the failing step in the user's words;
    every user-facing name (a reflected setting, a message key, a block name) greps to one definition;
    a bug fix is a few-line diff in one file.
11. **Ground truth forces extraction.**
    Pure arithmetic with an analytic answer is a named function with its own test, however small.
12. **Measure, and settle arguments with a counter-example.**
    When "it cannot be faster" is claimed, a small reference implementation decides.
    Bound: most code uses little of the CPU — measure first, then decide whether it matters.

Where the stance is risky: a long function can hide interleaved state (bound: principle 5);
a recurring pattern that is wrong spreads (bound: principle 4);
a hub type every file writes into is locality for the writer, not the reader (bound: group state by its owner);
long functions are where automated edits go wrong (bound: distinct names for neighbouring locals; review the diff, not the description).

### LLM-generated code

LLM assistants drift into scope and feature creep: code that is longer, more general and more layered than the problem needs.
Industry data reports more duplicated blocks and less refactoring since assistants spread (GitClear, 2025, vendor data),
and a controlled comparison found more code smells in LLM output than in reference solutions (Ghosh Paul et al., 2025).
The code stays reviewed and owned by its developer; the patterns `CLAUDE.md` §2 lists under "MUST NOT appear" are rejected whoever wrote them.

### Measuring it

`devtools/code-metrics.sh --help` explains every figure; `CLAUDE.md` §2 "Measured quality" gives the order in which alternatives are compared.
Cognitive complexity (Campbell, 2017) is summed over the project, never read as a per-function maximum, and split-invariant, so splitting a function cannot lower it;
it charges nesting and broken flow rather than length, so a long straight sequence costs little.
Reading volume replaces physical lines:
splitting one 600-column line into three named `const` locals adds lines and almost no tokens, and makes the line readable.
A comment is reading time like code: one that says in prose what the next lines say in C++ costs that time twice.
No metric predicts understandability alone:
the figures are computed after review, inform it and never replace it,
and a regression a change accepts is named, with its number, in the commit message.

## Specific Development Guidelines

These specific guidelines complement the [C++ Core Guidelines](https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines), providing direction in cases where there are multiple
coding approaches, that are not (yet) standardised, or slightly deviate from standard practices.

0. **Apply Common Sense, pay-it-forward, and the Golden Rule**: read and adhere to the [Code of Conduct](CODE_OF_CONDUCT.md) and the [Contribution Guidelines](CONTRIBUTING.md).

1. **Lean-Code Practices and the Boy-Scout Rule**
   - Adhere to the [C++ Core Guidelines](https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines) as your primary reference.
   - Embrace the **Boy Scout Rule**: _Leave the code leaner than you found it._

2. **Tool Mastery**: _"If your only tool is a hammer, all problems appear as nails."_
   - Avoid committing code that triggers compiler warnings: `-Werror` is activated, so a warning fails CI.
   - Cross-check your change with `devtools/code-metrics.sh` (before and after) and `devtools/naming-check`; they are indicators of where code can be simplified, and the CI bots post the same figures on your pull request.
   - Prioritise clarity over complexity: strive for code that's easy to read, efficient, and maintainable.
     You do not have to demonstrate the sophistication of your (meta-)programming capabilities, we just assume you are a good programmer.
   - Format with `./formatFiles.sh` (`--install` once fetches the pinned formatter versions); an IDE format-on-save with the repository's `.clang-format` keeps C++ in line between runs.

   _Non-compliance will lead to CI rejection and often reduces the likeliness of someone reviewing your high-level code concepts._

3. **Coding Best Practices**:
   - Minimise variable scope and delay definitions until usage.
   - Use `using` and `using namespace` sparingly and within limited scope.
   - Avoid variable shadowing; opt for unique expressive names.
   - Avoid `bool` function arguments in APIs -- they are often ambiguous and confusing.
   - Keep one sequence in one function, balancing length against an excessive number of minor, hard-to-follow functions (principle 5 above).

4. **The one C++ Standard**:
   - Use the latest C++ standard adopted by the project, considering the common denominator supported by CI's stable versions of GCC, Clang, and Emscripten.
   - Capitalise on the Standard Template Library (STL, (std)libc++). Opt for `<algorithm>`, `<numeric>`, `<ranges>`, etc., over custom solutions.
   - Favour `concepts` over `static_assert` for broader compile-time checks.

5. **Minimise 3rd-Party Library Dependencies**: except in the following cases:
   - they are crucial for core functionality (e.g. ZeroMQ or OpenSSL for network communication);
   - small libraries that can be internalised within the project;
   - they are pre-agreed upon, only used in a sub-project, and can be conditionally disabled.

6. **Effective Testing and Documentation:**
   - Ensure functional unit testing and meaningful code coverage; a `qa_` file runs its suites in `main` (`CLAUDE.md` §8).
   - Use `static_assert`s for compile-time verifications.
   - Provide usage examples to illustrate the library's simplified common core use alongside unit-tests that need to handle more complex and less common corner cases.
   - Accept both test-driven and example-driven development for API design.

7. **Naming Convention**: Please read and follow the [naming guidelines](CORE_NAMING_GUIDELINE.md).

8. **Structs vs Classes**: strongly prioritise functional programming where possible to mitigate performance bottlenecks
   and race conditions, especially in multi-threaded environments.
   - Favour public `structs` (aka. _aggregates_) over `classes` unless encapsulation of a non-public invariant or shared state (e.g. a HW resource) is required.
   - Prefix member variables that have a predominantly 'private' use-case with an underscore (\_).
   - The order of members is given in `CLAUDE.md` §4.

9. **Header-only Code and `const`**: where code lives (header or `.cpp`) is given in `CLAUDE.md` §2 "Placement".
   Use `const` and `constexpr` liberally, except when they impede performance (e.g. 'const' disabling move semantics).

10. **Contemporary C++ Practices:**
    - Don't overuse and reserve the use of `auto` where it enhances readability or is required.
    - Raw pointers
      - should be limited to specific use cases, must be non-owning, and marked as unsafe when outside those parameters.
      - prefer wrapping them into a RAII-enabled types.
      - are fine if they are used without pointer arithmetic
    - Pass by value or reference-to-const based on the data type size and context.
    - Exceptions should remain exceptional. Opt for alternatives like `std::optional<>` or `std::expected<>` for simpler stacks.
    - **Inheritance Is The Base Class of Evil** (Parent, 2013):
      strongly favour composition over inheritance. Avoid the latter unless RTTI is absolutely necessary.
      _N.B. This has a strong impact on the long-term evolution and maintainability of the project.
      Strongly Object-Oriented-Designs with prolific inheritance hierarchy make it difficult to near impossible to change designs without breaking existing code._
    - Use contemporary C++ features like enum classes, `override`, and `final` where appropriate.

## Code formatting

`./formatFiles.sh` applies the repository's formatters (clang-format, cmake-format, prettier, shellcheck's fixes, whitespace, black) at pinned versions,
so style is never a review topic. Two `.clang-format` choices need a reason:

1. **Unlimited column length.**
   A strict limit wraps code where the tool decides, not where the logic breaks, and pushes developers towards abbreviations or `auto` to fit.
   It is not a licence for long lines: a line that grows long is split into named `const` values, one idea each, not wrapped (`CLAUDE.md` §9);
   `code-metrics` counts lines over 160 characters.
   To force a break at a chosen place, end the line with an empty `//` comment.
2. **Column alignment.**
   Aligned types, names and values make declarations and initialisations read as a table, so patterns and discrepancies show at a glance.

## References

- M. E. Conway, "How Do Committees Invent?", _Datamation_, vol. 14, no. 4, pp. 28–31, 1968 ([paper](https://www.melconway.com/Home/Committees_Paper.html), [PDF](https://www.melconway.com/Home/pdf/committees.pdf)).
- C. Muratori, "Designing and Evaluating Reusable Components", GameTech, 2004 ([article](https://caseymuratori.com/blog_0024));
  "Semantic Compression", 2014 ([article](https://caseymuratori.com/blog_0015));
  "Complexity and Granularity", 2014 ([article](https://caseymuratori.com/blog_0016));
  "The Thirty-Million-Line Problem", talk, 2015 ([article](https://caseymuratori.com/blog_0031), 2018);
  "Where Does Bad Code Come From?", Molly Rocket, c. 2021 ([commentary by B. Christel](https://github.com/benchristel/benchristel.github.io/wiki/Commentary:WhereDoesBadCodeComeFrom));
  "The Only Unbreakable Law", Molly Rocket, 2022 ([video](https://www.youtube.com/watch?v=5IUj1EZwpJY));
  "'Clean' Code, Horrible Performance", 2023 ([article](https://www.computerenhance.com/p/clean-code-horrible-performance));
  "The Big OOPs: Anatomy of a Thirty-Five-Year Mistake", Better Software Conference, 2025 ([video](https://www.youtube.com/watch?v=wo84LFzx5nI), [article](https://www.computerenhance.com/p/the-big-oops-anatomy-of-a-thirty)).
- C. Muratori and R. C. Martin, written debate, 2023 ([repository](https://github.com/unclebob/cmuratori-discussion); the primary source for "operation first", not a recommendation of _Clean Code_).
- S. Metz, "The Wrong Abstraction", 2016 ([article](https://sandimetz.com/blog/2016/1/20/the-wrong-abstraction)).
- J. Lakos, _Large-Scale C++ Software Design_, Addison-Wesley, 1996; _Large-Scale C++ Volume I_, Addison-Wesley, 2019.
- J. Ousterhout, _A Philosophy of Software Design_, Yaknyam Press, 2018 (2nd ed. 2021) ([book page](https://web.stanford.edu/~ouster/cgi-bin/book.php)).
- J. Carmack, e-mail on inlined code, 2007, published 2014 ([text](http://number-none.com/blow/john_carmack_on_inlined_code.html)).
- D. E. Knuth, "Structured Programming with go to Statements", _ACM Computing Surveys_, vol. 6, no. 4, pp. 261–301, 1974 ([doi:10.1145/356635.356640](https://doi.org/10.1145/356635.356640)).
- R. Hickey, "Simple Made Easy", Strange Loop, 2011 ([talk](https://www.infoq.com/presentations/Simple-Made-Easy/)).
- C. Gross, "Locality of Behaviour", 2020 ([essay](https://htmx.org/essays/locality-of-behaviour/)).
- P. Naur, "Programming as Theory Building", _Microprocessing and Microprogramming_, vol. 15, no. 5, 1985 ([PDF](https://pages.cs.wisc.edu/~remzi/Naur.pdf)).
- G. A. Campbell, "Cognitive Complexity", SonarSource, 2017 ([PDF](https://www.sonarsource.com/docs/CognitiveComplexity.pdf)).
- F. P. Brooks, "No Silver Bullet — Essence and Accident in Software Engineering", 1986 ([PDF](http://worrydream.com/refs/Brooks_1986_-_No_Silver_Bullet.pdf)).
- S. Parent, "Inheritance Is The Base Class of Evil", GoingNative 2013 ([talk](https://www.youtube.com/watch?v=bIhUE5uUFOA)).
- M. Strathern, "'Improving ratings': audit in the British University system", _European Review_, vol. 5, no. 3, pp. 305–321, 1997 ([doi:10.1017/S1062798700002660](https://doi.org/10.1017/S1062798700002660)), after C. Goodhart, 1975.
- GitClear, "AI Copilot Code Quality: 2025 Look Back at 12 Months of Data", 2025 ([report](https://www.gitclear.com/ai_assistant_code_quality_2025_research); vendor data).
- A. Ghosh Paul, H. Zhu, I. Bayley, "Investigating the Smells of LLM Generated Code", 2025 ([arXiv:2510.03029](https://arxiv.org/abs/2510.03029)).
