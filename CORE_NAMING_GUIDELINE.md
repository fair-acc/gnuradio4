## Naming Rules and Guidelines
In software development, the right naming conventions bridge the gap between human linguistic patterns and logical code structures, turning abstract concepts into meaningful expressions and algorithmic intent. By aligning our codebase with familiar linguistic constructs, we aim to foster clarity, maintainability, and a more intuitive understanding of our project. Embracing lean and clean code principles, this guideline aims at a concise yet expressive naming. While we strive for consistency, we emphasize the spirit of clarity and meaning over rigid adherence when faced with exceptions.

This guideline is embedded into the larger [CORE_DEVELOPMENT_GUIDELINE.md](../blob/main/CORE_DEVELOPMENT_GUIDELINE.md) context of this project [README.md](../blob/main/README.md).
<details> <summary>click here for the TL;DR summary</summary>

## Naming Conventions: Quick Summary (TL;DR)

- **General Philosophy**: Strive for clarity and intuition. When in doubt, prioritize clarity over strict adherence to these rules.
- **Classes/Structs/Enums**: Use `UpperCase` like `Graph` or `SubGraph`.
- **Methods/Functions/Lambdas**: Use `lowerCase` such as `start()` or `create()`.
- **Fields**: Use `snake_case` for public fields like `is_valid`. Non-public fields start with `_`, e.g., `_initialised`.
- **Function Variables & Parameters**: Use `lowerCase` or discretion.
- **Template Parameters**: Types use `T` or `TSpecificName`. Non-types are `lowerCase` or context-specific.
- **Constants**: Use `kUppercase`, e.g., `kConstant`.
- **C++ Concepts**: Favor `UpperCase` like `PortLike` or `HasMethod`.
- **Namespaces**: Always `lowercase`.
- **Type Aliases**: Use `UpperCase`.
- **Files & Directories**: Reflect primary class/struct/template definition. Short, `lowercase` for directories.
- **Unit Tests**: Prefix with `qa_`.
- **Comments**: Be concise, focus on the 'why'. Use `/** ... */` for documentation, `//` for inline.
- **Macros**: Always `UPPERCASE`.
</details>

You can place this right at the beginning of your detailed guidelines so that developers can quickly familiarize themselves with the essentials before diving into the comprehensive guide.


### Preface Language
The keywords "MUST", "MUST NOT", "REQUIRED", "SHALL", "SHALL NOT", "SHOULD", "SHOULD NOT", "RECOMMENDED", "MAY", and "OPTIONAL" in this document are to be interpreted as described in [RFC 2119](https://datatracker.ietf.org/doc/html/rfc2119)

There are two distinct use cases to consider:

### 1. STL-style Library Usage
This pertains to ubiquitous, generic, domain-in-specific meta-programming which could, theoretically, be eventually included in the C++ standard, for instance, `type_list<>`:
* You SHOULD adhere strictly to C++ ISO guidelines and naming conventions.
* These definitions SHOULD be kept in separate headers and possibly within a distinct sub-project (though within the same repository) to facilitate potential future refactoring and outsourcing.

### 2. Domain-specific Library and User-code
For the domain-specific library usage and user-defined code, adhere to the following rules:

1. (templated-) **classes/structs** & **enums**: SHOULD use UpperCase in line with *proper nouns*. Examples: `Port`, `Block`, `Graph`, `SubGraph`, `Selector`.
    - enumerator: use UpperCase or lowerCase based on whether the enumerator is a *proper noun* (e.g. `Planet::Earth`, `Currency::Dollar`) or a *common noun* concept (e.g. `Color::red`, `Direction::north`) or use your judgment.
2. (templated-) **methods/functions/lambdas**: SHOULD use lowerCase in akin to *verbs*. Examples: `scheduler.start()`, `scheduler.stop()`, `filter::create()`, `computeMagnitude(...)`.
3. **fields** of classes/structs: MUST use lower_case. Examples: `temperature`, `filter`, `is_valid`, `sample_rate'.
    - place these at the top of the class definition for visibility (hidden state)
    - non-public/private fields MUST be prefixed by `_`. Examples: `_initialised`, `_cachedVector`
    - public API fields, especially if compile-time reflected, should strictly follow snake_case to simplify compatibility with [SigMF](https://github.com/sigmf/SigMF).
4. **function variables and parameters**: SHOULD use lowerCase or your own judgement. Examples: `sum`, `filter`, `isValid`
    - for functions/methods: should be defined as close to their initial use as possible
    - parameters may use `_` suffix in constructors to avoid name clashes with class field names (e.g. `User(std::string name_) : name(std::move(name_) {/*...*/}`)
5. **template parameters**:
    - **type parameters**: SHOULD use standard notations like `T`, `U`, `V` or prefix with `T` for more specific types. Examples: `template<typename TString>`, `template<typename TBlock>`
    - **non-type template parameters (NTTPs)**: SHOULD prefer lowerCase or use discretion based on context (e.g. `template<std::size_t bufferSize>`, `template<std::size_t N = std::dynamic_extent>`)
6. **constants**: SHOULD use kUppercase. Example: `kConstant`.
7. C++ **concepts**: SHOULD use UpperCase. Several alternatives are being considered. Among them are -- TO CHECK/VERIFY/ITERATE: (<-> must avoid collisions with template definitions)
    1. `IsBlock<T>`, `HasMethod<T>`, ...
    2. `PortLike<T>` (pro: `func(PortLike auto port)`) vs `IsBlock<T>` (pro: `requires (isBlock<T> && ...)`)
        * slightly alt: `PortLike<T>` && (`is_port_v<T>` || `isPort<T>`)
    3. `PortLike<T>` && `HasMethod<T>`
    4. `BlockConcept<T>`
    5. `XxxType` -> seems not to be favoured
    6. other...
8. **namespaces**: MUST use lowercase and descriptive scope names.
9. **using directives**:
    - **using type aliases**: SHOULD follow the class/struct rule (i.e., CamelCase), employ very sparingly and only when it substantially boosts clarity and readability and is recurrently used, like in concise concepts.
    - **using namespace**: MUST prefer the narrowest applicable scope __and__ if clarity dictates. Avoid broad usage to prevent namespace pollution.
10. **file names**: SHOULD reflect the primary class/struct/template definition contained within and follow their rules
11. **directory names**: SHOULD use short lowercase names roughly linked to namespace (sub-)scope names (keep hierarchy level low).
12. **unit tests**: MUST use `qa_` as a prefix and append the header file name of the class/struct/function-scope to be tested.
13. **comments**:
    - **class/struct/method comments**: if they cannot be trivially discerned from their name, brief implementation, etc. MUST be documented using the `/** ... */` or in-code `Doc<"..">` format.
    - **inline comments**: SHOULD use `//` for brief single-line comments directly above or alongside the code they explain. Please do not restate the obvious but explain the 'why' or complex 'how'.
    - **block comments**: MAY use `/* ... */` for multi-line comments. Please do not commit uncommented blocks of code but rather delete them.
    - **TODO comments**: MAY use `// TODO: ...` to highlight incomplete or temporary sections, detailing what's left or the reason for the temporary state.
    - *Note: if in doubt -- focus on lean- and clean code -- let the code speak for itself. Domain-specific aspects aside, the need for documentation can indicate poor, complex, or unintuitive designs.*
14. macros: MUST use UPPERCASE. Example: `ENABLE_REFLECTION`.
- convention adheres to established standards and serves as a cautionary highlight. Its 'shouting' nature signals developers that using such global/singleton patterns should be the exception rather than the norm.

Some example to provide a flavour of the naming scheme:

```cpp
// FILE: Block.hpp
namespace gr::merged {
template<typename T>
struct Block {}; // used-defined struct is proper noun (uppercase)
} // namespace gr

// FILE: Graph.hpp

namespace gr::merged {

template <typename T>
concept IsGraph = requires(T t) {
    { t.is_valid } -> std::same_as<bool>;
};

/** @brief Represents a simple graph structure for signals. (alt documentation)*/
template <typename TBlock>
class Graph : public Block<Graph<TBlock>> {
    bool               _initialised = true; // private member using `_` prefix

public:
    using Description = Doc<R"(
  @brief Represents a simple graph structure for signals. (preferred documentation)

  ... you may use markdown etc.
  This documentation can also be processed by the compiler w/o needing additional
  macro-processors or documentation-generating tools.
)">;
    static const int   kDefaultSize = 10;
    std::vector<TBlock> nodes;
    bool               is_valid; // field is public API using snake_case

    // constructor with parameter using `_` suffix to avoid name clashes
    Graph(std::vector<TBlock> nodes_) : nodes(std::move(nodes_)) {
        // TODO: initialization of other components
    }

    template <typename TFilter>
    void applyFilter(TFilter filter) { // templated method in camelCase
        for (auto& node : nodes) {
            // break up long method names semantically using namespace
            node = filter::create(node);
        }
        // ...
    }
};

} // namespace gr::merged
```

**Rationale for these Naming Conventions**

Discussion on naming conventions often slides into [bikeshedding](https://en.wikipedia.org/wiki/Law_of_triviality), resembling debates like British vs. American spelling, where trivial disagreements overshadow larger, more critical issues. Below is a rationale behind our choices:

1. **Aligning Code with Linguistics and Semantics**: domain-specific objects resemble *proper nouns* (typically capitalised in natural languages), actions mirror *verbs* (commonly lowercase), and variables mimic *common nouns* (generic and lowercase). This analogy helps facilitate an intuitive understanding of code semantics.

2. **English Vocabulary & Composition**: English predominantly uses single words, making up 80-90% of its vocabulary. Long or compound words often signal specific, non-generic meanings, placing them in the *proper noun* category. These should not solely be represented through class names but preferably described through functional attributes of the class or methods. In coding, such nuances should not be represented by class names but described through functional attributes, such as template parameters, non-type template parameters (NTTP), or namespace scope. This ensures the class’s attributes are queryable through meta-programming techniques rather than a specific type name check, enhancing composability.

3. **Research-Backed**: Research has shown that naming conventions significantly affect code readability and comprehension. These indicate that:
    - While familiar naming conventions can hasten reading, the priority in code review is accuracy, not speed. Over-reliance on familiarity may lead to missed nuances and errors.
    - CamelCase improves reading accuracy for both novice and experienced coders.
    - CamelCase does not significantly impact reading speed for experienced developers.
    - CamelCase engages more deliberate cognitive processing, aligning with Kahneman's 'System 2' responses.

   References: [Binkley et al., 2009](http://www.cs.loyola.edu/~binkley/papers/icpc09-clouds.pdf), [Kahneman & Tversky, 1979](https://www.uzh.ch/cmsssl/suz/dam/jcr:00000000-64a0-5b1c-0000-00003b7ec704/10.05-kahneman-tversky-79.pdf), [Kahneman, 2003](https://www2.econ.iastate.edu/tesfatsi/JudgementAndChoice.MappingBoundedRationality.DKahneman2003.pdf).
    <details><summary><b>'System 1' vs. 'System 2' TL;DR Summary</b></summary>

   Psychologist Daniel Kahneman introduced the concepts of 'System 1' and 'System 2' as two distinct modes of thinking in his groundbreaking work. Here's a quick rundown of these systems:

   ### 'System 1'

    - **Type**: Automatic, instinctive.
    - **Characteristics**:
        - Fast
        - Intuitive
        - Emotion-driven
        - Often operates subconsciously
    - **When it's active**:
        - Jumping to conclusions
        - Making gut reactions
        - Recognizing familiar patterns

   ### 'System 2'

    - **Type**: Deliberative, analytical.
    - **Characteristics**:
        - Slow
        - Logical
        - Requires effort
        - Conscious thinking involved
    - **When it's active**:
        - Solving complex math problems
        - Making deliberate choices
        - Evaluating evidence

   The distinction between 'System 1' and 'System 2' thinking is not just theoretical; it has practical applications in numerous fields, including software development. For a deeper dive into these concepts and their implications, consider reading Kahneman's book "[Thinking, Fast and Slow](https://www.amazon.com/Daniel-Kahneman/dp/0374533555)".
    </details>

4. **Leveraging Namespaces in C++**: for class or method meanings that are domain, context, or scope-specific, C++'s 'namespaces' are invaluable. They articulate this scope concisely, reducing the need for extended compound names. For example, `gr::algorithm::window::Type::Hann` or `gr::algorithm::window::create(Type windowFunction, std::size_t n, ...)`.

While multiple naming conventions have their merits, our aim is clarity, simplicity and consistency. We're confident that our chosen conventions best achieve this, sidestepping protracted, non-scientific, and unproductive debates over preferences.