#ifndef GRAPH_PROTOTYPE_CLOCK_SOURCE_HPP
#define GRAPH_PROTOTYPE_CLOCK_SOURCE_HPP

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <iostream>
#include <node.hpp>
#include <optional>
#include <random>
#include <reflection.hpp>
#include <tag.hpp>
#include <thread>
#include <queue>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include "../unit-test/tag_monitors.hpp"

namespace fair::graph::sources {

// optional shortening
template<typename T, fair::meta::fixed_string description = "", typename... Arguments>
using A = Annotated<T, description, Arguments...>;

template<typename T, typename ClockSourceType = std::chrono::system_clock, bool basicPeriodAlgorithm = true>
struct ClockSource : public node<ClockSource<T, ClockSourceType>, BlockingIO, Doc<R""(
ClockSource Documentation -- add here
)"">> {
    std::chrono::time_point<ClockSourceType> nextTimePoint = ClockSourceType::now();
    //
    OUT<T>             out;
    std::vector<tag_t> tags{};
    std::size_t        next_tag{ 0 };
    //
    A<std::uint32_t, "n_samples_max", Visible, Doc<"0: unlimited">>              n_samples_max = 1024;
    std::uint32_t                                                                n_samples_produced{ 0 };
    A<float, "avg. sample rate", Visible>                                        sample_rate = 1000.f;
    A<std::uint32_t, "chunk_size", Visible, Doc<"number of samples per update">> chunk_size  = 100;

    void
    settings_changed(const property_map & /*old_settings*/, const property_map & /*new_settings*/) {
        nextTimePoint = ClockSourceType::now();
    }

    work_return_status_t
    process_bulk(PublishableSpan auto &output) noexcept {
        if (n_samples_max > 0 && n_samples_produced >= n_samples_max) {
            output.publish(0_UZ);
            return work_return_status_t::DONE;
        }

        // sleep until next update period -- the following call blocks
        std::this_thread::sleep_until(nextTimePoint);
        const auto writableSamples = static_cast<std::uint32_t>(output.size());
        if (writableSamples < chunk_size) {
            output.publish(0_UZ);
            return work_return_status_t::INSUFFICIENT_OUTPUT_ITEMS;
        }

        const auto          remaining_samples  = static_cast<std::uint32_t>(n_samples_max - n_samples_produced);
        const std::uint32_t limit              = std::min(writableSamples, remaining_samples);
        const std::uint32_t n_available        = std::min(limit, static_cast<std::uint32_t>(chunk_size.value));

        std::uint32_t       samples_to_produce = n_available;
        while (next_tag < tags.size() && tags[next_tag].index <= static_cast<std::make_signed_t<std::size_t>>(n_samples_produced + n_available)) {
            tag_test::print_tag(tags[next_tag], fmt::format("{}::process_bulk(...)\t publish tag at  {:6}", this->name, n_samples_produced));
            tag_t &out_tag     = this->output_tags()[0];
            out_tag            = tags[next_tag];
            out_tag.index      = tags[next_tag].index - static_cast<std::make_signed_t<std::size_t>>(n_samples_produced);
            samples_to_produce = static_cast<std::uint32_t>(static_cast<std::uint32_t>(tags[next_tag].index) - n_samples_produced);
            this->forward_tags();
            next_tag++;
        }
        samples_to_produce = std::min(samples_to_produce, static_cast<std::uint32_t>(n_samples_max.value));

        output.publish(samples_to_produce);
        n_samples_produced += samples_to_produce;

        if constexpr (basicPeriodAlgorithm) {
            const auto updatePeriod = std::chrono::microseconds(static_cast<long>(1e6f * static_cast<float>(chunk_size) / sample_rate));
            nextTimePoint += updatePeriod;
        } else {
            const auto updatePeriod = std::chrono::microseconds(static_cast<long>(1e6f * static_cast<float>(chunk_size) / sample_rate));
            // verify the actual rate
            const auto actual_elapsed_time   = std::chrono::duration_cast<std::chrono::microseconds>(ClockSourceType::now() - nextTimePoint + updatePeriod).count();
            const auto expected_elapsed_time = 1e6f * static_cast<float>(chunk_size) / sample_rate;

            // adjust the next update period
            const float ratio = static_cast<float>(actual_elapsed_time) / expected_elapsed_time;
            nextTimePoint += std::chrono::microseconds(static_cast<long>(static_cast<float>(updatePeriod.count()) * ratio));
        }

        return work_return_status_t::OK;
    }
};

} // namespace fair::graph::sources

ENABLE_REFLECTION_FOR_TEMPLATE_FULL((typename T, typename ClockSourceType), (fair::graph::sources::ClockSource<T, ClockSourceType>), out, n_samples_max, chunk_size, sample_rate);

namespace fair::graph::sources {
static_assert(HasProcessBulkFunction<ClockSource<float>>);
} // namespace fair::graph::sources

#endif // GRAPH_PROTOTYPE_CLOCK_SOURCE_HPP
