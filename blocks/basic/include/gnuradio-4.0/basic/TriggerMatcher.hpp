#ifndef GNURADIO_TRIGGERMATCHER_HPP
#define GNURADIO_TRIGGERMATCHER_HPP

namespace gr::basic {

enum class TriggerMatchResult {
    Matching,    ///< Start a new dataset
    NotMatching, ///< Finish dataset
    Ignore       ///< Ignore tag
};

/**
 * Used for testing whether a tag should trigger data acquisition.
 *
 * For the 'Triggered' (data window) and 'Snapshot' (single sample) acquisition modes:
 * Stateless predicate to check whether a tag matches the trigger criteria.
 *
 * @code
 * auto matcher = [](const auto &tag) {
 *     const auto isTrigger = ...check if tag is trigger...;
 *     return isTrigger ? TriggerMatchResult::Matching : TriggerMatchResult::Ignore;
 * };
 * @endcode
 *
 * For the 'Multiplexed' acquisition mode: Possibly stateful object checking all incoming tags to control which data should be sent
 * to the listener.
 *
 * A new dataset is started when the matcher returns @c Start or @c StopAndStart.
 * A dataset is closed and sent when @c Stop or @StopAndStart is returned.
 *
 * For the multiplexed case, the matcher might be stateful and can rely on being called with each incoming tag exactly once, in the order they arrive.
 *
 * Example:
 *
 * @code
 * // matcher observing three possible tag values, "green", "yellow", "red".
 * // starting a dataset when seeing "green", stopping on "red", starting a new dataset on "yellow"
 * struct ColorMatcher {
 *     TriggerMatcherResult operator()(const Tag &tag) {
 *         if (tag == green || tag == yellow) {
 *             return TriggerMatchResult::Matching;
 *         }
 *         if (tag == red) {
 *             return TriggerMatchResult::NotMatching;
 *         }
 *
 *         return TriggerMatchResult::Ignore;
 *     }
 * };
 * @endcode
 *
 * @see TriggerMatchResult
 */
template<typename T>
concept TriggerMatcher = requires(T matcher, Tag tag) {
    { matcher(tag) } -> std::convertible_to<TriggerMatchResult>;
};

} // namespace gr::basic

#endif // GNURADIO_TRIGGERMATCHER_HPP
