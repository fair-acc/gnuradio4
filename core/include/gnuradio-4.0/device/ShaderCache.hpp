#ifndef GNURADIO_SHADER_CACHE_HPP
#define GNURADIO_SHADER_CACHE_HPP

#include <cstddef>
#include <functional>
#include <list>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>

namespace gr::device {

/**
 * @brief Thread-safe LRU cache mapping shader source (by hash) to compiled program handles.
 *
 * Backend-agnostic: the handle type THandle is opaque (GLuint for GLSL, pipeline ID for WGSL).
 * The caller provides a compile function; the cache invokes it on miss.
 *
 * @example
 * ShaderCache<unsigned int> cache(64);  // max 64 entries
 * auto handle = cache.getOrCompile(glslSource, [&](std::string_view src) {
 *     return gl.compileOrGetCached(src);  // returns std::expected<unsigned int, std::string>
 * });
 */
template<typename THandle>
struct ShaderCache {
    std::size_t _maxEntries;

    mutable std::mutex _mtx;

    // LRU list: front = most recently used, back = least recently used
    using LruList = std::list<std::size_t>; // stores source hashes in LRU order
    using LruIter = typename LruList::iterator;

    struct Entry {
        THandle handle;
        LruIter lruPos;
    };

    LruList                                _lruOrder;
    std::unordered_map<std::size_t, Entry> _entries;
    std::function<void(THandle)>           _onEvict; // optional cleanup callback

    explicit ShaderCache(std::size_t maxEntries = 128, std::function<void(THandle)> onEvict = {}) : _maxEntries(maxEntries), _onEvict(std::move(onEvict)) {}

    /// look up or compile a shader. CompileFn: (std::string_view) → std::optional<THandle>
    template<typename CompileFn>
    std::optional<THandle> getOrCompile(std::string_view source, CompileFn&& compile) {
        auto hash = std::hash<std::string_view>{}(source);

        std::scoped_lock lk(_mtx);

        auto it = _entries.find(hash);
        if (it != _entries.end()) {
            // cache hit — move to front of LRU
            _lruOrder.splice(_lruOrder.begin(), _lruOrder, it->second.lruPos);
            return it->second.handle;
        }

        // cache miss — compile
        auto result = compile(source);
        if (!result) {
            return std::nullopt;
        }

        // evict LRU if at capacity
        while (_entries.size() >= _maxEntries && !_lruOrder.empty()) {
            auto evictHash = _lruOrder.back();
            _lruOrder.pop_back();
            auto evictIt = _entries.find(evictHash);
            if (evictIt != _entries.end()) {
                if (_onEvict) {
                    _onEvict(evictIt->second.handle);
                }
                _entries.erase(evictIt);
            }
        }

        // insert new entry at front of LRU
        _lruOrder.push_front(hash);
        _entries[hash] = {*result, _lruOrder.begin()};
        return *result;
    }

    /// invalidate a specific shader (e.g. on settings change)
    void invalidate(std::string_view source) {
        auto             hash = std::hash<std::string_view>{}(source);
        std::scoped_lock lk(_mtx);
        auto             it = _entries.find(hash);
        if (it != _entries.end()) {
            _lruOrder.erase(it->second.lruPos);
            if (_onEvict) {
                _onEvict(it->second.handle);
            }
            _entries.erase(it);
        }
    }

    /// invalidate all cached shaders
    void clear() {
        std::scoped_lock lk(_mtx);
        if (_onEvict) {
            for (auto& [_, entry] : _entries) {
                _onEvict(entry.handle);
            }
        }
        _entries.clear();
        _lruOrder.clear();
    }

    [[nodiscard]] std::size_t size() const noexcept {
        std::scoped_lock lk(_mtx);
        return _entries.size();
    }
};

} // namespace gr::device

#endif // GNURADIO_SHADER_CACHE_HPP
