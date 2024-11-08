#ifndef GNURADIO_SEQUENCE_HPP
#define GNURADIO_SEQUENCE_HPP

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <limits>
#include <memory>
#include <ranges>
#include <vector>

#include <fmt/format.h>

namespace gr {

#ifndef forceinline
// use this for hot-spots only <-> may bloat code size, not fit into cache and
// consequently slow down execution
#define forceinline inline __attribute__((always_inline))
#endif

#ifdef __cpp_lib_hardware_interference_size
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
 * Also attempts to be more efficient with regards to false sharing by adding padding
 * around the volatile field.
 */
class Sequence {
    alignas(hardware_destructive_interference_size) std::atomic<std::size_t> _fieldsValue{};

public:
    Sequence(const Sequence&)       = delete;
    Sequence(const Sequence&&)      = delete;
    void operator=(const Sequence&) = delete;

    explicit Sequence(std::size_t initialValue = kInitialCursorValue) noexcept : _fieldsValue(initialValue) {}

    [[nodiscard]] forceinline std::size_t value() const noexcept { return std::atomic_load_explicit(&_fieldsValue, std::memory_order_acquire); }
    forceinline void                      setValue(const std::size_t value) noexcept { std::atomic_store_explicit(&_fieldsValue, value, std::memory_order_release); }

    [[nodiscard]] forceinline bool compareAndSet(std::size_t expectedSequence, std::size_t nextSequence) noexcept {
        // atomically set the value to the given updated value if the current value == the
        // expected value (true, otherwise folse).
        return std::atomic_compare_exchange_strong(&_fieldsValue, &expectedSequence, nextSequence);
    }

    [[maybe_unused]] forceinline std::size_t incrementAndGet() noexcept { return std::atomic_fetch_add(&_fieldsValue, 1L) + 1L; }
    [[nodiscard]] forceinline std::size_t addAndGet(std::size_t value) noexcept { return std::atomic_fetch_add(&_fieldsValue, value) + value; }
    [[nodiscard]] forceinline std::size_t subAndGet(std::size_t value) noexcept { return std::atomic_fetch_sub(&_fieldsValue, value) - value; }
    void                                  wait(std::size_t oldValue) const noexcept { atomic_wait_explicit(&_fieldsValue, oldValue, std::memory_order_acquire); }
    void                                  notify_all() noexcept { _fieldsValue.notify_all(); }
};

namespace detail {

/**
 * Get the minimum sequence from an array of Sequences.
 *
 * \param sequences sequences to compare.
 * \param minimum an initial default minimum.  If the array is empty this value will
 * returned. \returns the minimum sequence found or lon.MaxValue if the array is empty.
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

#include <fmt/core.h>
#include <fmt/ostream.h>

template<>
struct fmt::formatter<gr::Sequence> {
    template<typename ParseContext>
    constexpr auto parse(ParseContext& ctx) {
        return ctx.begin();
    }

    template<typename FormatContext>
    auto format(gr::Sequence const& value, FormatContext& ctx) const {
        return fmt::format_to(ctx.out(), "{}", value.value());
    }
};

namespace gr {
inline std::ostream& operator<<(std::ostream& os, const Sequence& v) { return os << fmt::format("{}", v.value()); }
} // namespace gr

#endif // GNURADIO_SEQUENCE_HPP
