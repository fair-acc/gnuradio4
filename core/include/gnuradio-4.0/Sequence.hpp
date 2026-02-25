#ifndef GNURADIO_SEQUENCE_HPP
#define GNURADIO_SEQUENCE_HPP

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <ranges>
#include <vector>

#include <format>

#include <gnuradio-4.0/AtomicRef.hpp>

namespace gr {

#ifndef forceinline
// use this for hot-spots only <-> may bloat code size, not fit into cache and
// consequently slow down execution
#define forceinline inline __attribute__((always_inline))
#endif

#if defined(__APPLE__) && defined(__aarch64__)
// Apple Silicon (M1–M4) uses 128-byte L2 cache lines
inline constexpr std::size_t hardware_destructive_interference_size  = 128;
inline constexpr std::size_t hardware_constructive_interference_size = 128;
#elif defined(__cpp_lib_hardware_interference_size)
using std::hardware_constructive_interference_size;
using std::hardware_destructive_interference_size;
#else
inline constexpr std::size_t hardware_destructive_interference_size  = 64;
inline constexpr std::size_t hardware_constructive_interference_size = 64;
#endif
static constexpr const std::size_t kInitialCursorValue = 0L;

/**
 * Concurrent sequence class used for tracking the progress of the ring buffer and event
 * processors. Support a number of concurrent operations including CAS and order writes.
 * Also avoids false sharing by adding padding cacheline-padding around the volatile field.
 */
class alignas(hardware_destructive_interference_size) Sequence {
    mutable std::size_t _fieldsValue{kInitialCursorValue};

public:
    Sequence(const Sequence&)       = delete;
    Sequence(const Sequence&&)      = delete;
    void operator=(const Sequence&) = delete;

    Sequence() = default;
    explicit Sequence(std::size_t v) noexcept { gr::atomic_ref(_fieldsValue).store_release(v); }

    [[nodiscard]] forceinline std::size_t value() const noexcept { return gr::atomic_ref(_fieldsValue).load_acquire(); }
    forceinline void                      setValue(const std::size_t value) noexcept { gr::atomic_ref(_fieldsValue).store_release(value); }

    [[nodiscard]] forceinline bool compareAndSet(std::size_t expectedSequence, std::size_t nextSequence) noexcept {
        // atomically set the value to the given updated value if the current value == the expected value (true, otherwise folse).
        return gr::atomic_ref(_fieldsValue).compare_exchange(expectedSequence, nextSequence);
    }

    [[maybe_unused]] forceinline std::size_t incrementAndGet() noexcept { return gr::atomic_ref(_fieldsValue).fetch_add(1UZ) + 1UZ; }
    [[nodiscard]] forceinline std::size_t addAndGet(std::size_t increment) noexcept { return gr::atomic_ref(_fieldsValue).fetch_add(increment) + increment; }
    [[nodiscard]] forceinline std::size_t subAndGet(std::size_t decrement) noexcept { return gr::atomic_ref(_fieldsValue).fetch_sub(decrement) - decrement; }
    void                                  wait(std::size_t oldValue) const noexcept { gr::atomic_ref(_fieldsValue).wait(oldValue); }
    void                                  notify_all() noexcept { gr::atomic_ref(_fieldsValue).notify_all(); }
};

namespace detail {

/**
 * Get the minimum sequence from an array of Sequences.
 *
 * \param sequences sequences to compare.
 * \param minimum the initial default minimum. If the array is empty, this value will be returned.
 * \returns the minimum sequence found or lon.MaxValue if the array is empty.
 */
inline std::size_t getMinimumSequence(const std::vector<std::shared_ptr<Sequence>>& sequences, std::size_t minimum = std::numeric_limits<std::size_t>::max()) noexcept {
    // Note that calls to getMinimumSequence get rather expensive with sequences.size() because
    // each Sequence lives on its own cache line. Also, this is no reasonable loop for vectorization.
    for (const auto& s : sequences) {
        const std::size_t v = s->value();
        if (v < minimum) {
            minimum = v;
        }
    }
    return minimum;
}

// TODO: Revisit this code once libc++ adds support for `std::atomic<std::shared_ptr<std::vector<std::shared_ptr<Sequence>>>>`.
// Currently, suppressing deprecation warnings for `std::atomic_load_explicit` and `std::atomic_compare_exchange_weak` methods.
// Note: While `std::atomic<std::shared_ptr<std::vector<std::shared_ptr<Sequence>>>>` is compatible with GCC, it is not yet supported by libc++.
// This workaround is necessary to maintain compatibility and avoid deprecation warnings in GCC. For more details, see the following example implementation:
// https://godbolt.org/z/xxWbs659o
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif

inline void addSequences(std::shared_ptr<std::vector<std::shared_ptr<Sequence>>>& sequences, const Sequence& cursor, const std::vector<std::shared_ptr<Sequence>>& sequencesToAdd) {
    std::size_t                                             cursorSequence;
    std::shared_ptr<std::vector<std::shared_ptr<Sequence>>> updatedSequences;
    std::shared_ptr<std::vector<std::shared_ptr<Sequence>>> currentSequences;

    do {
        currentSequences = std::atomic_load_explicit(&sequences, std::memory_order_acquire);
        updatedSequences = std::make_shared<std::vector<std::shared_ptr<Sequence>>>(currentSequences->size() + sequencesToAdd.size());

#if not defined(_LIBCPP_VERSION)
        std::ranges::copy(currentSequences->begin(), currentSequences->end(), updatedSequences->begin());
#else
        std::copy(currentSequences->begin(), currentSequences->end(), updatedSequences->begin());
#endif

        cursorSequence = cursor.value();

        auto index = currentSequences->size();
        for (auto&& sequence : sequencesToAdd) {
            sequence->setValue(cursorSequence);
            (*updatedSequences)[index] = sequence;
            index++;
        }
    } while (!std::atomic_compare_exchange_weak(&sequences, &currentSequences, updatedSequences)); // xTODO: explicit memory order

    cursorSequence = cursor.value();

    for (auto&& sequence : sequencesToAdd) {
        sequence->setValue(cursorSequence);
    }
}

inline bool removeSequence(std::shared_ptr<std::vector<std::shared_ptr<Sequence>>>& sequences, const std::shared_ptr<Sequence>& sequence) {
    std::uint32_t                                           numToRemove;
    std::shared_ptr<std::vector<std::shared_ptr<Sequence>>> oldSequences;
    std::shared_ptr<std::vector<std::shared_ptr<Sequence>>> newSequences;

    do {
        oldSequences = std::atomic_load_explicit(&sequences, std::memory_order_acquire);
#if not defined(_LIBCPP_VERSION)
        numToRemove = static_cast<std::uint32_t>(std::ranges::count(*oldSequences, sequence)); // specifically uses identity
#else
        numToRemove = static_cast<std::uint32_t>(std::count((*oldSequences).begin(), (*oldSequences).end(), sequence)); // specifically uses identity
#endif
        if (numToRemove == 0) {
            break;
        }

        auto oldSize = static_cast<std::uint32_t>(oldSequences->size());
        newSequences = std::make_shared<std::vector<std::shared_ptr<Sequence>>>(oldSize - numToRemove);

        for (auto i = 0U, pos = 0U; i < oldSize; ++i) {
            const auto& testSequence = (*oldSequences)[i];
            if (sequence != testSequence) {
                (*newSequences)[pos] = testSequence;
                pos++;
            }
        }
    } while (!std::atomic_compare_exchange_weak(&sequences, &oldSequences, newSequences));

    return numToRemove != 0;
}
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

} // namespace detail

} // namespace gr

#include <gnuradio-4.0/meta/formatter.hpp>

template<>
struct std::formatter<gr::Sequence> {
    template<typename ParseContext>
    constexpr auto parse(ParseContext& ctx) {
        return ctx.begin();
    }

    template<typename FormatContext>
    auto format(gr::Sequence const& value, FormatContext& ctx) const {
        return std::format_to(ctx.out(), "{}", value.value());
    }
};

namespace gr {
inline std::ostream& operator<<(std::ostream& os, const Sequence& v) { return os << std::format("{}", v.value()); }
} // namespace gr

#ifdef forceinline
#undef forceinline
#endif

#endif // GNURADIO_SEQUENCE_HPP
