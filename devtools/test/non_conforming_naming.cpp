// example that is not conforming with the naming convention
#include <vector>

namespace Graphics { // non-conforming namespace (should be lowercase)

template<typename TGraph>
struct GRAPH {                           // non-conforming struct name (should be UpperCamelCase)
    static const int defaultSize = 2048; // non-conforming constant (should start with k and be CamelCase)
    bool             Initialized;        // non-conforming public field (should be snake_case)

    void
    Initialize() { // non-conforming method name (should be lowerCamelCase)
        Initialized = true;
    }
};

template<typename T>
class graph { // non-conforming class name (should be UpperCamelCase)
private:
    bool privateVariable; // non-conforming private field
protected:
    bool protectedVariable;             // non-conforming protected field
    bool protected_snake_case_variable; // non-conforming protected field
public:
    bool public_variable; // conforming public field
public:
    void
    ComputeData() {} // non-conforming method name (should be lowerCamelCase)
};

template<typename T>
concept Drawable = requires(T x) {
    { x.Draw() } -> std::same_as<void>; // non-conforming method name in concept
};

using vertex = int; // non-conforming type alias (should be UpperCamelCase)

} // namespace Graphics

int
main() {
    Graphics::GRAPH<double> main_graph;
    main_graph.Initialize();
    return 0;
}
