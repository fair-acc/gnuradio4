#ifndef GNURADIO_TAGMONITORS_HPP
#define GNURADIO_TAGMONITORS_HPP

#include <limits>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/Tag.hpp>
#include <gnuradio-4.0/reflection.hpp>

namespace gr::testing {

enum class ProcessFunction {
    USE_PROCESS_BULK     = 0, ///
    USE_PROCESS_ONE      = 1, ///
    USE_PROCESS_ONE_SIMD = 2  ///
};

inline constexpr void print_tag(const Tag& tag, std::string_view prefix = {}) noexcept {
    if (tag.map.empty()) {
        fmt::print("{} @index= {}: map: {{ <empty map> }}\n", prefix, tag.index);
        return;
    }
    fmt::print("{} @index= {}: map: {{ {} }}\n", prefix, tag.index, fmt::join(tag.map, ", "));
}

template<typename MapType>
inline constexpr void map_diff_report(const MapType& map1, const MapType& map2, const std::string& name1, const std::string& name2, const std::optional<std::string>& ignoreKey = std::nullopt) {
    for (const auto& [key, value] : map1) {
        if (ignoreKey && key == *ignoreKey) {
            continue; // skip this key
        }
        const auto it = map2.find(key);
        if (it == map2.end()) {
            fmt::print("    key '{}' is present in {} but not in {}\n", key, name1, name2);
        } else if (it->second != value) {
            fmt::print("    key '{}' has different values ('{}' vs '{}')\n", key, value, it->second);
        }
    }
}

template<typename IterType>
inline constexpr void mismatch_report(const IterType& mismatchedTag1, const IterType& mismatchedTag2, const IterType& tags1_begin, const std::optional<std::string>& ignoreKey = std::nullopt) {
    const auto index = static_cast<size_t>(std::distance(tags1_begin, mismatchedTag1));
    fmt::print("mismatch at index {}", index);
    if (mismatchedTag1->index != mismatchedTag2->index) {
        fmt::print("  - different index: {} vs {}\n", mismatchedTag1->index, mismatchedTag2->index);
    }

    if (mismatchedTag1->map != mismatchedTag2->map) {
        fmt::print("  - different map content:\n");
        map_diff_report(mismatchedTag1->map, mismatchedTag2->map, "the first map", "the second", ignoreKey);
        map_diff_report(mismatchedTag2->map, mismatchedTag1->map, "the second map", "the first", ignoreKey);
    }
}

inline constexpr bool equal_tag_lists(const std::vector<Tag>& tags1, const std::vector<Tag>& tags2, const std::optional<std::string>& ignoreKey = std::nullopt) {
    if (tags1.size() != tags2.size()) {
        fmt::println("vectors have different sizes ({} vs {})\n", tags1.size(), tags2.size());
        return false;
    }

    auto customComparator = [&ignoreKey](const Tag& tag1, const Tag& tag2) {
        if (ignoreKey) {
            // make a copy of the maps to compare without the ignored key
            auto map1 = tag1.map;
            auto map2 = tag2.map;
            map1.erase(*ignoreKey);
            map2.erase(*ignoreKey);
            return map1 == map2;
        }
        return tag1 == tag2; // Use Tag's equality operator
    };

    auto [mismatchedTag1, mismatchedTag2] = std::mismatch(tags1.begin(), tags1.end(), tags2.begin(), customComparator);
    if (mismatchedTag1 != tags1.end()) {
        mismatch_report(mismatchedTag1, mismatchedTag2, tags1.begin(), ignoreKey);
        return false;
    }
    return true;
}

template<typename T, ProcessFunction UseProcessVariant = ProcessFunction::USE_PROCESS_BULK>
struct TagSource : public Block<TagSource<T, UseProcessVariant>> {
    PortOut<T> out;

    bool           repeat_tags = false;      // if true tags are repeated from the beginning. Example: Given the tag indices {1, 3, 5}, the output tag indices would be: 1, 3, 5, 6, 8, 10, ...
    std::vector<T> values{};                 // if values are set it works like repeated source. Example: values = { 1, 2, 3 }; output: 1,2,3,1,2,3... `mark_tag` is ignored in this case.
    gr::Size_t     n_samples_max{1024};      // if 0 -> infinite samples
    gr::Size_t     n_samples_produced{0ULL}; // for infinite samples the counter wraps around back to 0, _tagIndex = 0, _valueIndex = 0
    float          sample_rate     = 1000.0f;
    std::string    signal_name     = "unknown signal";
    std::string    signal_unit     = "unknown unit";
    float          signal_min      = std::numeric_limits<float>::lowest();
    float          signal_max      = std::numeric_limits<float>::max();
    bool           verbose_console = false;
    bool           mark_tag        = true; // true: mark tagged samples with '1' or '0' otherwise. false: [0, 1, 2, ..., ], if values is not empty mark_tag is ignored

    std::vector<Tag> _tags{};        // It is expected that Tag.index is in ascending order
    std::size_t      _tagIndex{0};   // current index in tags array
    std::size_t      _valueIndex{0}; // current index in values array

    void start() {
        n_samples_produced = 0U;
        _valueIndex        = 0U;
        _tagIndex          = 0U;
        if (_tags.size() > 1) {
            bool isAscending = std::ranges::is_sorted(_tags, [](const Tag& lhs, const Tag& rhs) { return lhs.index < rhs.index; });
            if (!isAscending) {
                using namespace gr::message;
                this->emitErrorMessage("error()", Error("The input tags should be ascending by index."));
            }
        }
    }

    T processOne(std::size_t offset) noexcept
    requires(UseProcessVariant == ProcessFunction::USE_PROCESS_ONE)
    {
        const auto [tagGenerated, tagRepeatStarted] = generateTag("processOne(...)", offset);
        n_samples_produced++;
        if (!isInfinite() && n_samples_produced >= n_samples_max) {
            this->requestStop();
        }

        if (isInfinite() && tagRepeatStarted) {
            n_samples_produced = 0U;
        }

        if (!values.empty()) {
            if (_valueIndex == values.size()) {
                _valueIndex = 0;
            }
            T currentValue = values[_valueIndex];
            _valueIndex++;
            return currentValue;
        }
        return mark_tag ? (tagGenerated ? static_cast<T>(1) : static_cast<T>(0)) : static_cast<T>(n_samples_produced);
    }

    work::Status processBulk(PublishableSpan auto& output) noexcept
    requires(UseProcessVariant == ProcessFunction::USE_PROCESS_BULK)
    {
        const auto [tagGenerated, tagRepeatStarted] = generateTag("processBulk(...)");
        const auto nSamplesRemainder                = getNProducedSamplesRemainder();

        gr::Size_t nextTagIn = 1U;
        if (isInfinite() && tagRepeatStarted) {
            nextTagIn = 1; // just publish last tag and then start from the beginning
        } else {
            if (_tagIndex < _tags.size()) {
                if (static_cast<gr::Size_t>(_tags[_tagIndex].index) > nSamplesRemainder) {
                    nextTagIn = static_cast<gr::Size_t>(_tags[_tagIndex].index) - nSamplesRemainder;
                }
            } else {
                nextTagIn = isInfinite() ? static_cast<gr::Size_t>(output.size()) : n_samples_max - n_samples_produced;
            }
        }

        const std::size_t nSamples = isInfinite() || n_samples_produced < n_samples_max ? std::min(static_cast<std::size_t>(std::max(1U, nextTagIn)), output.size()) : 0UZ; // '0UZ' -> DONE, produced enough samples

        if (!values.empty()) {
            for (std::size_t i = 0; i < nSamples; ++i) {
                if (_valueIndex == values.size()) {
                    _valueIndex = 0;
                }
                output[i] = values[_valueIndex];
                _valueIndex++;
            }
        } else {
            if (mark_tag) {
                output[0] = tagGenerated ? static_cast<T>(1) : static_cast<T>(0);
            } else {
                for (std::size_t i = 0; i < nSamples; ++i) {
                    output[i] = static_cast<T>(n_samples_produced + i);
                }
            }
        }

        if (isInfinite() && tagRepeatStarted) {
            n_samples_produced = 0U;
        } else {
            n_samples_produced += static_cast<gr::Size_t>(nSamples);
        }
        output.publish(nSamples);
        return !isInfinite() && n_samples_produced >= n_samples_max ? work::Status::DONE : work::Status::OK;
    }

private:
    [[nodiscard]] auto generateTag(std::string_view processFunctionName, std::size_t offset = 0) {
        struct {
            bool tagGenerated     = false;
            bool tagRepeatStarted = false;
        } result;

        const auto nSamplesRemainder = getNProducedSamplesRemainder();
        if (_tagIndex < _tags.size() && static_cast<gr::Size_t>(_tags[_tagIndex].index) <= nSamplesRemainder) {
            if (verbose_console) {
                print_tag(_tags[_tagIndex], fmt::format("{}::{}\t publish tag at  {:6}", this->name.value, processFunctionName, n_samples_produced));
            }
            out.publishTag(_tags[_tagIndex].map, static_cast<Tag::signed_index_type>(offset)); // indices > 0 write tags in the future ... handle with care
            this->_outputTagsChanged = true;
            _tagIndex++;
            if (repeat_tags && _tagIndex == _tags.size()) {
                _tagIndex               = 0;
                result.tagRepeatStarted = true;
            }
            result.tagGenerated = true;
            return result;
        }
        return result;
    }

    [[nodiscard]] gr::Size_t getNProducedSamplesRemainder() const { //
        return repeat_tags && !_tags.empty() && !isInfinite() ? n_samples_produced % static_cast<gr::Size_t>(_tags.back().index + 1) : n_samples_produced;
    }

    [[nodiscard]] bool isInfinite() const { return n_samples_max == 0U; }
};

template<typename T, ProcessFunction UseProcessVariant>
struct TagMonitor : public Block<TagMonitor<T, UseProcessVariant>> {
    PortIn<T>  in;
    PortOut<T> out;

    // Currently, UncertainValue is not supported in Settings. For now, just remove the error part and store only values.
    using TDataTypeStored = UncertainValueType_t<T>;

    std::vector<TDataTypeStored> samples{};
    gr::Size_t                   n_samples_expected{0};
    gr::Size_t                   n_samples_produced{0}; // for infinite samples the counter wraps around back to 0
    float                        sample_rate = 1000.0f;
    std::string                  signal_name;
    bool                         log_tags        = true;
    bool                         log_samples     = true;
    bool                         verbose_console = false;

    std::vector<Tag> _tags{};

    void start() {
        if (verbose_console) {
            fmt::println("started TagMonitor {} aka. '{}'", this->unique_name, this->name);
        }
        samples.clear();
        if (log_samples) {
            samples.reserve(std::max(0UZ, static_cast<std::size_t>(n_samples_expected)));
        }
        _tags.clear();
    }

    constexpr T processOne(const T& input) noexcept
    requires(UseProcessVariant == ProcessFunction::USE_PROCESS_ONE)
    {
        if (this->input_tags_present()) {
            const Tag& tag = this->mergedInputTag();
            if (verbose_console) {
                print_tag(tag, fmt::format("{}::processOne(...)\t received tag at {:6}", this->name, n_samples_produced));
            }
            if (log_tags) {
                _tags.emplace_back(n_samples_produced, tag.map);
            }
        }
        if (log_samples) {
            if constexpr (UncertainValueLike<T>) {
                samples.emplace_back(input.value);
            } else {
                samples.emplace_back(input);
            }
        }
        n_samples_produced++;
        return input;
    }

    template<gr::meta::t_or_simd<T> V>
    [[nodiscard]] constexpr V processOne(const V& input) noexcept // to note: the SIMD-version does not support adding tags mid-way since this is chunked at V::size()
    requires(UseProcessVariant == ProcessFunction::USE_PROCESS_ONE_SIMD)
    {
        if (this->input_tags_present()) {
            const Tag& tag = this->mergedInputTag();
            if (verbose_console) {
                print_tag(tag, fmt::format("{}::processOne(...)\t received tag at {:6}", this->name, n_samples_produced));
            }
            if (log_tags) {
                _tags.emplace_back(n_samples_produced, tag.map);
            }
        }
        if (log_samples) {
            if constexpr (gr::meta::any_simd<V>) {
                alignas(stdx::memory_alignment_v<stdx::native_simd<T>>) std::array<T, V::size()> mem = {};
                input.copy_to(&mem[0], stdx::vector_aligned);
                if constexpr (UncertainValueLike<T>) {
                    for (const auto& item : mem) {
                        samples.emplace_back(item.value);
                    }
                } else {
                    samples.insert(samples.end(), mem.begin(), mem.end());
                }
            } else {
                if constexpr (UncertainValueLike<T>) {
                    samples.emplace_back(input.value);
                } else {
                    samples.emplace_back(input);
                }
            }
        }
        if constexpr (gr::meta::any_simd<V>) {
            n_samples_produced += static_cast<gr::Size_t>(V::size());
        } else {
            n_samples_produced++;
        }
        return input;
    }

    constexpr work::Status processBulk(std::span<const T> input, std::span<T> output) noexcept
    requires(UseProcessVariant == ProcessFunction::USE_PROCESS_BULK)
    {
        if (this->input_tags_present()) {
            const Tag& tag = this->mergedInputTag();
            if (verbose_console) {
                print_tag(tag, fmt::format("{}::processBulk(...{}, ...{})\t received tag at {:6}", this->name, input.size(), output.size(), n_samples_produced));
            }
            if (log_tags) {
                _tags.emplace_back(n_samples_produced, tag.map);
            }
        }

        if (log_samples) {
            if constexpr (UncertainValueLike<T>) {
                for (const auto& item : input) {
                    samples.emplace_back(item.value);
                }
            } else {
                samples.insert(samples.end(), input.begin(), input.end());
            }
        }

        n_samples_produced += static_cast<gr::Size_t>(input.size());
        std::memcpy(output.data(), input.data(), input.size() * sizeof(T));

        return work::Status::OK;
    }
};

template<typename T, ProcessFunction UseProcessVariant>
struct TagSink : public Block<TagSink<T, UseProcessVariant>> {
    using ClockSourceType = std::chrono::system_clock;
    PortIn<T> in;

    // Currently, UncertainValue is not supported in Settings. For now, just remove the error part and store only values.
    using TDataTypeStored = UncertainValueType_t<T>;

    std::vector<TDataTypeStored> samples{};
    gr::Size_t                   n_samples_expected{0};
    gr::Size_t                   n_samples_produced{0}; // for infinite samples the counter wraps around back to 0
    float                        sample_rate = 1000.0f;
    std::string                  signal_name;
    bool                         log_tags        = true;
    bool                         log_samples     = true;
    bool                         verbose_console = false;

    std::vector<Tag> _tags{};

    void start() {
        if (verbose_console) {
            fmt::println("started sink {} aka. '{}'", this->unique_name, this->name);
        }
        samples.clear();
        if (log_samples) {
            samples.reserve(std::max(0UZ, static_cast<std::size_t>(n_samples_expected)));
        }
        _tags.clear();
    }

    void stop() {
        if (verbose_console) {
            fmt::println("stopped sink {} aka. '{}'", this->unique_name, this->name);
        }
    }

    constexpr void processOne(const T& input) noexcept // N.B. non-SIMD since we need a sample-by-sample accurate tag detection
    requires(UseProcessVariant == ProcessFunction::USE_PROCESS_ONE)
    {
        if (this->input_tags_present()) {
            const Tag& tag = this->mergedInputTag();
            if (verbose_console) {
                print_tag(tag, fmt::format("{}::processOne(...1)    \t received tag at {:6}", this->name, n_samples_produced));
            }
            if (log_tags) {
                _tags.emplace_back(n_samples_produced, tag.map);
            }
        }
        if (log_samples) {
            if constexpr (UncertainValueLike<T>) {
                samples.emplace_back(input.value);
            } else {
                samples.emplace_back(input);
            }
        }
        n_samples_produced++;
        if (n_samples_expected > 0 && n_samples_produced >= n_samples_expected) {
            this->requestStop();
        }
    }

    // template<gr::meta::t_or_simd<T> V>
    constexpr work::Status processBulk(std::span<const T> input) noexcept
    requires(UseProcessVariant == ProcessFunction::USE_PROCESS_BULK)
    {
        if (this->input_tags_present()) {
            const Tag& tag = this->mergedInputTag();
            if (verbose_console) {
                print_tag(tag, fmt::format("{}::processBulk(...{})\t received tag at {:6}", this->name, input.size(), n_samples_produced));
            }
            if (log_tags) {
                _tags.emplace_back(n_samples_produced, tag.map);
            }
        }
        if (log_samples) {
            if constexpr (UncertainValueLike<T>) {
                for (const auto& item : input) {
                    samples.emplace_back(item.value);
                }
            } else {
                samples.insert(samples.end(), input.begin(), input.end());
            }
        }
        n_samples_produced += static_cast<gr::Size_t>(input.size());
        return n_samples_expected > 0 && n_samples_produced >= n_samples_expected ? work::Status::DONE : work::Status::OK;
    }
};

} // namespace gr::testing

ENABLE_REFLECTION_FOR_TEMPLATE_FULL((typename T, gr::testing::ProcessFunction b), (gr::testing::TagSource<T, b>), out, n_samples_max, sample_rate, signal_name, signal_unit, signal_min, signal_max, verbose_console, mark_tag, values, repeat_tags);
ENABLE_REFLECTION_FOR_TEMPLATE_FULL((typename T, gr::testing::ProcessFunction b), (gr::testing::TagMonitor<T, b>), in, out, n_samples_expected, sample_rate, signal_name, n_samples_produced, log_tags, log_samples, verbose_console, samples);
ENABLE_REFLECTION_FOR_TEMPLATE_FULL((typename T, gr::testing::ProcessFunction b), (gr::testing::TagSink<T, b>), in, n_samples_expected, sample_rate, signal_name, n_samples_produced, log_tags, log_samples, verbose_console, samples);

auto registerTagSource  = gr::registerBlock<gr::testing::TagSource, gr::testing::ProcessFunction::USE_PROCESS_ONE, float, double>(gr::globalBlockRegistry()) | gr::registerBlock<gr::testing::TagSource, gr::testing::ProcessFunction::USE_PROCESS_BULK, float, double>(gr::globalBlockRegistry());
auto registerTagMonitor = gr::registerBlock<gr::testing::TagMonitor, gr::testing::ProcessFunction::USE_PROCESS_ONE, float, double>(gr::globalBlockRegistry()) | gr::registerBlock<gr::testing::TagMonitor, gr::testing::ProcessFunction::USE_PROCESS_BULK, float, double>(gr::globalBlockRegistry());
auto registerTagSink    = gr::registerBlock<gr::testing::TagSink, gr::testing::ProcessFunction::USE_PROCESS_ONE, float, double>(gr::globalBlockRegistry()) | gr::registerBlock<gr::testing::TagSink, gr::testing::ProcessFunction::USE_PROCESS_BULK, float, double>(gr::globalBlockRegistry());

#endif // GNURADIO_TAGMONITORS_HPP
