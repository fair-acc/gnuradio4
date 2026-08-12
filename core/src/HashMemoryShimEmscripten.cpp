#include <cstddef>
#include <cstdint>

// Prebuilt gnuradio-core.a (LLVM 21 libc++ ABI ne2101xx) references
// std::__2::__hash_memory as an exported runtime symbol. Some Emscripten
// sysroot builds of pic libc++ omit that definition. Provide a weak fallback
// so MAIN_MODULE hosts linking WHOLE_ARCHIVE core still link cleanly.
namespace std {
inline namespace __2 {

__attribute__((pure)) std::size_t __hash_memory(const void* ptr, std::size_t size) noexcept {
    // FNV-1a — sufficient stand-in when the sysroot libc++ lacks the symbol.
    std::size_t hash = sizeof(std::size_t) == 8 ? static_cast<std::size_t>(14695981039346656037ull) : static_cast<std::size_t>(2166136261u);
    const auto* p    = static_cast<const unsigned char*>(ptr);
    for (std::size_t i = 0; i < size; ++i) {
        hash ^= p[i];
        hash *= sizeof(std::size_t) == 8 ? static_cast<std::size_t>(1099511628211ull) : static_cast<std::size_t>(16777619u);
    }
    return hash;
}

} // namespace __2
} // namespace std
