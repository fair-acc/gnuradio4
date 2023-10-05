#ifndef GNURADIO_SEQUENCE_HPP
#define GNURADIO_SEQUENCE_HPP

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <limits>
#include <memory>
#include <ranges>
#include <vector>

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
static constexpr const std::make_signed_t<std::size_t> kInitialCursorValue = -1L;

/**
 * Concurrent sequence class used for tracking the progress of the ring buffer and event
 * processors. Support a number of concurrent operations including CAS and order writes.
 * Also attempts to be more efficient with regards to false sharing by adding padding
 * around the volatile field.
 */
class Sequence {
public:
    using signed_index_type = std::make_signed_t<std::size_t>;

private:
    alignas(hardware_destructive_interference_size) std::atomic<signed_index_type> _fieldsValue{};

public:
    Sequence(const Sequence &)  = delete;
    Sequence(const Sequence &&) = delete;
    void
    operator=(const Sequence &)
            = delete;

    explicit Sequence(signed_index_type initialValue = kInitialCursorValue) noexcept : _fieldsValue(initialValue) {}

    [[nodiscard]] forceinline signed_index_type
    value() const noexcept {
        return std::atomic_load_explicit(&_fieldsValue, std::memory_order_acquire);
    }

    forceinline void
    setValue(const signed_index_type value) noexcept {
        std::atomic_store_explicit(&_fieldsValue, value, std::memory_order_release);
    }

    [[nodiscard]] forceinline bool
    compareAndSet(signed_index_type expectedSequence, signed_index_type nextSequence) noexcept {
        // atomically set the value to the given updated value if the current value == the
        // expected value (true, otherwise folse).
        return std::atomic_compare_exchange_strong(&_fieldsValue, &expectedSequence, nextSequence);
    }

    [[nodiscard]] forceinline signed_index_type
    incrementAndGet() noexcept {
        return std::atomic_fetch_add(&_fieldsValue, 1L) + 1L;
    }

    [[nodiscard]] forceinline signed_index_type
    addAndGet(signed_index_type value) noexcept {
        return std::atomic_fetch_add(&_fieldsValue, value) + value;
    }

    void
    wait(signed_index_type oldValue) const noexcept {
        atomic_wait_explicit(&_fieldsValue, oldValue, std::memory_order_acquire);
    }

    void
    notify_all() noexcept {
        _fieldsValue.notify_all();
    }
};

namespace detail {

using signed_index_type = Sequence::signed_index_type;

/**
 * Get the minimum sequence from an array of Sequences.
 *
 * \param sequences sequences to compare.
 * \param minimum an initial default minimum.  If the array is empty this value will
 * returned. \returns the minimum sequence found or lon.MaxValue if the array is empty.
 */
inline signed_index_type
getMinimumSequence(const std::vector<std::shared_ptr<Sequence>> &sequences, signed_index_type minimum = std::numeric_limits<signed_index_type>::max()) noexcept {
    // Note that calls to getMinimumSequence get rather expensive with sequences.size() because
    // each Sequence lives on its own cache line. Also, this is no reasonable loop for vectorization.
    for (const auto &s : sequences) {
        const signed_index_type v = s->value();
        if (v < minimum) {
            minimum = v;
        }
    }
    return minimum;
}

inline void
addSequences(std::shared_ptr<std::vector<std::shared_ptr<Sequence>>> &sequences, const Sequence &cursor, const std::vector<std::shared_ptr<Sequence>> &sequencesToAdd) {
    signed_index_type                                       cursorSequence;
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

        auto index     = currentSequences->size();
        for (auto &&sequence : sequencesToAdd) {
            sequence->setValue(cursorSequence);
            (*updatedSequences)[index] = sequence;
            index++;
        }
    } while (!std::atomic_compare_exchange_weak(&sequences, &currentSequences, updatedSequences)); // xTODO: explicit memory order

    cursorSequence = cursor.value();

    for (auto &&sequence : sequencesToAdd) {
        sequence->setValue(cursorSequence);
    }
}

inline bool
removeSequence(std::shared_ptr<std::vector<std::shared_ptr<Sequence>>> &sequences, const std::shared_ptr<Sequence> &sequence) {
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
            const auto &testSequence = (*oldSequences)[i];
            if (sequence != testSequence) {
                (*newSequences)[pos] = testSequence;
                pos++;
            }
        }
    } while (!std::atomic_compare_exchange_weak(&sequences, &oldSequences, newSequences));

    return numToRemove != 0;
}

} // namespace detail

} // namespace gr

#ifdef FMT_FORMAT_H_
#include <fmt/core.h>
#include <fmt/ostream.h>

template<>
struct fmt::formatter<gr::Sequence> {
    template<typename ParseContext>
    constexpr auto
    parse(ParseContext &ctx) {
        return ctx.begin();
    }

    template<typename FormatContext>
    auto
    format(gr::Sequence const &value, FormatContext &ctx) {
        return fmt::format_to(ctx.out(), "{}", value.value());
    }
};

namespace gr {
inline std::ostream &
operator<<(std::ostream &os, const Sequence &v) {
    return os << fmt::format("{}", v.value());
}
} // namespace gr

#endif // FMT_FORMAT_H_

#endif // GNURADIO_SEQUENCE_HPP
