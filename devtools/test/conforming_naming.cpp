// example that is conforming with the naming convention
#include <vector>

namespace graphics {

template<typename TWindow>
struct Window {
    static const int kDefaultSize = 1024; // conforming constant
    bool             is_initialized;      // conforming public field

    void
    initialize() {
        is_initialized = true;
    }
};

template<typename T>
class Graph {
private:
    bool _privateVariable; // conforming private field
protected:
    bool _protectedVariable; // conforming protected field
public:
    bool public_variable; // conforming public field

    void
    computeData() {} // conforming method
};

template<typename T>
concept Drawable = requires(T x) {
    { x.draw() } -> std::same_as<void>;
};

using Vertex = int; // Conforming type alias

} // namespace graphics

int
main() {
    graphics::Window<double> mainWindow;
    mainWindow.initialize();
    return 0;
}
