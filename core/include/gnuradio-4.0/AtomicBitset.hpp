#ifndef GNURADIO_ATOMICBITSET_HPP
#define GNURADIO_ATOMICBITSET_HPP

#include <cassert>
#include <vector>

#include <gnuradio-4.0/AtomicRef.hpp>

#ifndef forceinline
// use this for hot-spots only <-> may bloat code size, not fit into cache and consequently slow down execution
#define forceinline inline __attribute__((always_inline))
#endif

namespace gr {
/*
 * `AtomicBitset` is a lock-free, thread-safe bitset.
 * It allows for efficient and thread-safe manipulation of individual bits.
 * For bulk set or reset atomic operation is guaranteed per individual bit (word).
 */
template<std::size_t Size = std::dynamic_extent>
class AtomicBitset {
    static_assert(Size > 0, "Size must be greater than 0");
    static constexpr bool isSizeDynamic = Size == std::dynamic_extent;

    static constexpr std::size_t _bitsPerWord  = sizeof(size_t) * 8UZ;
    static constexpr std::size_t _nStaticWords = isSizeDynamic ? 1UZ : (Size + _bitsPerWord - 1UZ) / _bitsPerWord;

    using DynamicArrayType = std::vector<std::size_t>;
    using StaticArrayType  = std::array<std::size_t, _nStaticWords>;
    using ArrayType        = std::conditional_t<isSizeDynamic, DynamicArrayType, StaticArrayType>;

    std::size_t       _size = Size;
    mutable ArrayType _bits;

public:
    AtomicBitset()
    requires(!isSizeDynamic)
    {
        for (auto& word : _bits) {
            gr::atomic_ref(word).store_relaxed(0UZ);
        }
    }

    explicit AtomicBitset(std::size_t size = 0UZ)
    requires(isSizeDynamic)
        : _size(size), _bits((size + _bitsPerWord - 1UZ) / _bitsPerWord) {
        for (auto& word : _bits) {
            gr::atomic_ref(word).store_relaxed(0UZ);
        }
    }

    void set(std::size_t bitPosition, bool value) {
        assert(bitPosition < _size);
        const std::size_t wordIndex = bitPosition / _bitsPerWord;
        const std::size_t bitIndex  = bitPosition % _bitsPerWord;
        const std::size_t mask      = 1UZ << bitIndex;

        std::size_t oldBits;
        std::size_t newBits;
        do {
            oldBits = gr::atomic_ref(_bits[wordIndex]).load_relaxed();
            newBits = value ? (oldBits | mask) : (oldBits & ~mask);
        } while (!gr::atomic_ref(_bits[wordIndex]).compare_exchange(oldBits, newBits));
    }

    void set(std::size_t begin, std::size_t end, bool value) {
        assert(begin <= end && end <= _size); // [begin, end)

        std::size_t       beginWord   = begin / _bitsPerWord;
        std::size_t       endWord     = end / _bitsPerWord;
        const std::size_t beginOffset = begin % _bitsPerWord;
        const std::size_t endOffset   = end % _bitsPerWord;

        if (begin == end) {
            return;
        } else if (beginWord == endWord) {
            // the range is within a single word
            setBitsInWord(beginWord, beginOffset, endOffset, value);
        } else {
            // leading bits in the first word
            if (beginOffset != 0) {
                setBitsInWord(beginWord, beginOffset, _bitsPerWord, value);
                beginWord++;
            }
            // trailing bits in the last word
            if (endOffset != 0) {
                setBitsInWord(endWord, 0, endOffset, value);
            }
            endWord--;
            // whole words in the middle
            for (std::size_t wordIndex = beginWord; wordIndex <= endWord; ++wordIndex) {
                setFullWord(wordIndex, value);
            }
        }
    }

    void set(std::size_t bitPosition) { set(bitPosition, true); }

    void set(std::size_t begin, std::size_t end) { set(begin, end, true); }

    void reset(std::size_t bitPosition) { set(bitPosition, false); }

    void reset(std::size_t begin, std::size_t end) { set(begin, end, false); }

    bool test(std::size_t bitPosition) const {
        assert(bitPosition < _size);
        const std::size_t wordIndex = bitPosition / _bitsPerWord;
        const std::size_t bitIndex  = bitPosition % _bitsPerWord;
#if defined(_WIN32)
        const std::size_t mask = 1ULL << bitIndex;
#else
        const std::size_t mask = 1UL << bitIndex;
#endif

        return (gr::atomic_ref(_bits[wordIndex]).load_acquire() & mask) != 0;
    }

    [[nodiscard]] constexpr std::size_t size() const { return _size; }

private:
    void setBitsInWord(std::size_t wordIndex, std::size_t begin, std::size_t end, bool value) {
        assert(begin < end && end <= _bitsPerWord);
        const std::size_t mask = (end == _bitsPerWord) ? ~((1UZ << begin) - 1) : ((1UZ << end) - 1) & ~((1UZ << begin) - 1);
        std::size_t       oldBits;
        std::size_t       newBits;
        do {
            oldBits = gr::atomic_ref(_bits[wordIndex]).load_relaxed();
            newBits = value ? (oldBits | mask) : (oldBits & ~mask);
        } while (!gr::atomic_ref(_bits[wordIndex]).compare_exchange(oldBits, newBits));
    }

    forceinline void setFullWord(std::size_t wordIndex, bool value) { gr::atomic_ref(_bits[wordIndex]).store_release(value ? ~0UZ : 0UZ); }
};

} // namespace gr

#endif // GNURADIO_ATOMICBITSET_HPP
