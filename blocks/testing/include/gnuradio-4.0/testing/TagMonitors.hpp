#ifndef GNURADIO_TAGMONITORS_HPP
#define GNURADIO_TAGMONITORS_HPP

#include <limits>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/Tag.hpp>
#include <gnuradio-4.0/meta/formatter.hpp>
#include <gnuradio-4.0/meta/reflection.hpp>

namespace gr::testing {

enum class ProcessFunction {
    USE_PROCESS_BULK = 0, ///
    USE_PROCESS_ONE  = 1  ///
};

inline constexpr void print_tag(const Tag& tag, std::string_view prefix = {}) noexcept {
    if (tag.map.empty()) {
        std::print("{} @index= {}: map: {{ <empty map> }}\n", prefix, tag.index);
        return;
    }
    std::print("{} @index= {}: map: {{ {} }}\n", prefix, tag.index, gr::join(tag.map, ", "));
}

template<typename MapType>
inline constexpr void map_diff_report(const MapType& map1, const MapType& map2, const std::string& name1, const std::string& name2, const std::optional<std::vector<std::string>>& ignoreKeys = std::nullopt) {
    const auto skipKey = [&](const auto& key) { return ignoreKeys != std::nullopt && std::ranges::find(ignoreKeys.value(), key) != ignoreKeys.value().end(); };

    for (const auto& [key, value] : map1) {
        if (skipKey(key)) {
            continue;
        }
        const auto it = map2.find(key);
        if (it == map2.end()) {
            std::print("    key '{}' is present in {} but not in {}\n", key, name1, name2);
        } else if (it->second != value) {
            std::print("    key '{}' has different values ('{}' vs '{}')\n", key, value, it->second);
        }
    }

    for (const auto& [key, value] : map2) {
        if (skipKey(key) || map1.contains(key)) {
            continue;
        }
        std::print("    key '{}' is present in {} but not in {}\n", key, name2, name1);
    }
}

template<typename IterType>
inline constexpr void mismatch_report(const IterType& mismatchedTag1, const IterType& mismatchedTag2, const IterType& tags1_begin, const std::optional<std::vector<std::string>>& ignoreKeys = std::nullopt) {
    const auto index = static_cast<std::size_t>(std::distance(tags1_begin, mismatchedTag1));
    std::print("mismatch at index {}", index);
    if (mismatchedTag1->index != mismatchedTag2->index) {
        std::print("  - different index: {} vs {}\n", mismatchedTag1->index, mismatchedTag2->index);
    }

    if (mismatchedTag1->map != mismatchedTag2->map) {
        std::print("  - different map content:\n");
        map_diff_report(mismatchedTag1->map, mismatchedTag2->map, "the first map", "the second", ignoreKeys);
        map_diff_report(mismatchedTag2->map, mismatchedTag1->map, "the second map", "the first", ignoreKeys);
    }
}

inline constexpr bool equal_tag_lists(const std::vector<Tag>& tags1, const std::vector<Tag>& tags2, const std::optional<std::vector<std::string>>& ignoreKeys = std::nullopt) {
    if (tags1.size() != tags2.size()) {
        std::println("vectors have different sizes ({} vs {})\n", tags1.size(), tags2.size());
        return false;
    }

    auto customComparator = [&ignoreKeys](const Tag& tag1, const Tag& tag2) {
        if (ignoreKeys != std::nullopt && !ignoreKeys.value().empty()) {
            // make a copy of the maps to compare without the ignored key
            auto map1 = tag1.map;
            auto map2 = tag2.map;
            for (const auto& ignoreKey : ignoreKeys.value()) {
                map1.erase(ignoreKey);
                map2.erase(ignoreKey);
            }
            return map1 == map2;
        }
        return tag1 == tag2; // Use Tag's equality operator
    };

    auto [mismatchedTag1, mismatchedTag2] = std::mismatch(tags1.begin(), tags1.end(), tags2.begin(), customComparator);
    if (mismatchedTag1 != tags1.end()) {
        mismatch_report(mismatchedTag1, mismatchedTag2, tags1.begin(), ignoreKeys);
        return false;
    }
    return true;
}

GR_REGISTER_BLOCK("gr::testing::TagSource", gr::testing::TagSource, ([T], gr::testing::ProcessFunction::USE_PROCESS_ONE), [ float, double ])

template<typename T, ProcessFunction UseProcessVariant = ProcessFunction::USE_PROCESS_BULK>
struct TagSource : Block<TagSource<T, UseProcessVariant>> {
    PortOut<T> out;

    // settings
    bool           repeat_tags = false; // if true tags are repeated from the beginning. Example: Given the tag indices {1, 3, 5}, the output tag indices would be: 1, 3, 5, 6, 8, 10, ...
    std::vector<T> values{};            // if values are set it works like repeated source. Example: values = { 1, 2, 3 }; output: 1,2,3,1,2,3... `mark_tag` is ignored in this case.
    gr::Size_t     n_samples_max{1024}; // if 0 -> infinite samples
    float          sample_rate     = 1000.0f;
    std::string    signal_name     = "unknown signal";
    std::string    signal_unit     = "unknown unit";
    float          signal_min      = std::numeric_limits<float>::lowest();
    float          signal_max      = std::numeric_limits<float>::max();
    bool           verbose_console = false;
    bool           mark_tag        = true; // true: mark tagged samples with '1' or '0' otherwise. false: [0, 1, 2, ..., ], if values is not empty mark_tag is ignored

    GR_MAKE_REFLECTABLE(TagSource, out, n_samples_max, sample_rate, signal_name, signal_unit, signal_min, signal_max, verbose_console, mark_tag, values, repeat_tags);

    std::vector<Tag> _tags{};                 // It is expected that Tag.index is in ascending order
    std::size_t      _tagIndex{0};            // current index in tags array
    std::size_t      _valueIndex{0};          // current index in values array
    gr::Size_t       _nSamplesProduced{0ULL}; // for infinite samples the counter wraps around back to 0, _tagIndex = 0, _valueIndex = 0

    std::function<void(const Tag&)> _tagCallback{}; // optional tag callback

    void start() {
        _nSamplesProduced = 0U;
        _valueIndex       = 0U;
        _tagIndex         = 0U;
        if (_tags.size() > 1) {
            bool isAscending = std::ranges::is_sorted(_tags, [](const Tag& lhs, const Tag& rhs) { return lhs.index < rhs.index; });
            if (!isAscending) {
                using namespace gr::message;
                this->emitErrorMessage("error()", Error("The input tags should be ascending by index."));
            }
        }
    }

    T processOne() noexcept
    requires(UseProcessVariant == ProcessFunction::USE_PROCESS_ONE)
    {
        const auto [tagGenerated, tagRepeatStarted] = generateTag(                                //
            [this](const auto& map, std::size_t tagOffset) { this->publishTag(map, tagOffset); }, //
            "processOne(...)");

        _nSamplesProduced++;
        if (!isInfinite() && _nSamplesProduced >= n_samples_max) {
            this->requestStop();
        }

        if (isInfinite() && tagRepeatStarted) {
            _nSamplesProduced = 0U;
        }

        if (!values.empty()) {
            if (_valueIndex == values.size()) {
                _valueIndex = 0;
            }
            T currentValue = values[_valueIndex];
            _valueIndex++;
            return currentValue;
        }
        return mark_tag ? (tagGenerated ? static_cast<T>(1) : static_cast<T>(0)) : static_cast<T>(_nSamplesProduced);
    }

    work::Status processBulk(OutputSpanLike auto& outSpan) noexcept
    requires(UseProcessVariant == ProcessFunction::USE_PROCESS_BULK)
    {
        const auto [tagGenerated, tagRepeatStarted] = generateTag(                                //
            [&outSpan](const auto& map, std::size_t offset) { outSpan.publishTag(map, offset); }, //
            "processBulk(...)");

        const auto nSamplesRemainder = getNProducedSamplesRemainder();

        gr::Size_t nextTagIn = 1U;
        if (isInfinite() && tagRepeatStarted) {
            nextTagIn = 1; // just publish last tag and then start from the beginning
        } else {
            if (_tagIndex < _tags.size()) {
                if (static_cast<gr::Size_t>(_tags[_tagIndex].index) > nSamplesRemainder) {
                    nextTagIn = static_cast<gr::Size_t>(_tags[_tagIndex].index) - nSamplesRemainder;
                    nextTagIn = std::min(nextTagIn, n_samples_max - _nSamplesProduced);
                }
            } else {
                nextTagIn = isInfinite() ? static_cast<gr::Size_t>(outSpan.size()) : n_samples_max - _nSamplesProduced;
            }
        }

        const std::size_t nSamples = isInfinite() || _nSamplesProduced < n_samples_max ? std::min(static_cast<std::size_t>(std::max(1U, nextTagIn)), outSpan.size()) : 0UZ; // '0UZ' -> DONE, produced enough samples

        if (!values.empty()) {
            for (std::size_t i = 0; i < nSamples; ++i) {
                if (_valueIndex == values.size()) {
                    _valueIndex = 0;
                }
                outSpan[i] = values[_valueIndex];
                _valueIndex++;
            }
        } else {
            if (mark_tag) {
                outSpan[0] = tagGenerated ? static_cast<T>(1) : static_cast<T>(0);
            } else {
                for (std::size_t i = 0; i < nSamples; ++i) {
                    outSpan[i] = static_cast<T>(_nSamplesProduced + i);
                }
            }
        }

        if (isInfinite() && tagRepeatStarted) {
            _nSamplesProduced = 0U;
        } else {
            _nSamplesProduced += static_cast<gr::Size_t>(nSamples);
        }
        outSpan.publish(nSamples);
        return !isInfinite() && _nSamplesProduced >= n_samples_max ? work::Status::DONE : work::Status::OK;
    }

private:
    template<typename TPublishTagFunction>
    [[nodiscard]] auto generateTag(TPublishTagFunction&& publishTagFn, std::string_view processFunctionName) {
        struct {
            bool tagGenerated     = false;
            bool tagRepeatStarted = false;
        } result;

        const auto nSamplesRemainder = getNProducedSamplesRemainder();
        if (_tagIndex < _tags.size() && static_cast<gr::Size_t>(_tags[_tagIndex].index) <= nSamplesRemainder) {
            if (verbose_console) {
                print_tag(_tags[_tagIndex], std::format("{}::{}\t publish tag at  {:6}", this->name.value, processFunctionName, _nSamplesProduced));
            }
            publishTagFn(_tags[_tagIndex].map, 0UZ);
            if (_tagCallback) {
                _tagCallback(_tags[_tagIndex]);
            }
            this->_outputTagsChanged = true;
            _tagIndex++;
            if (repeat_tags && _tagIndex == _tags.size()) {
                _tagIndex               = 0;
                result.tagRepeatStarted = true;
            }
            result.tagGenerated = true;
        }
        return result;
    }

    [[nodiscard]] gr::Size_t getNProducedSamplesRemainder() const { //
        return repeat_tags && !_tags.empty() && !isInfinite() ? _nSamplesProduced % static_cast<gr::Size_t>(_tags.back().index + 1) : _nSamplesProduced;
    }

    [[nodiscard]] bool isInfinite() const { return n_samples_max == 0U; }
};

GR_REGISTER_BLOCK("gr::testing::TagMonitor", gr::testing::TagMonitor, ([T], gr::testing::ProcessFunction::USE_PROCESS_ONE), [ float, double ])

template<typename T, ProcessFunction UseProcessVariant>
struct TagMonitor : public Block<TagMonitor<T, UseProcessVariant>> {
    PortIn<T>  in;
    PortOut<T> out;

    // settings
    gr::Size_t  n_samples_expected{0};
    float       sample_rate = 1000.0f;
    std::string signal_name;
    bool        log_tags        = true;
    bool        log_samples     = true;
    bool        verbose_console = false;

    GR_MAKE_REFLECTABLE(TagMonitor, in, out, n_samples_expected, sample_rate, signal_name, log_tags, log_samples, verbose_console);

    std::vector<T>   _samples;
    std::vector<Tag> _tags;
    gr::Size_t       _nSamplesProduced{0}; // for infinite samples the counter wraps around back to 0

    std::function<void(const Tag&)> _tagCallback{}; // optional tag callback

    void start() {
        if (verbose_console) {
            std::println("started TagMonitor {} aka. '{}'", this->unique_name, this->name);
        }
        _samples.clear();
        if (log_samples) {
            _samples.reserve(std::max(0UZ, static_cast<std::size_t>(n_samples_expected)));
        }
        _tags.clear();
    }

    constexpr T processOne(const T& input) noexcept
    requires(UseProcessVariant == ProcessFunction::USE_PROCESS_ONE)
    {
        if (this->inputTagsPresent()) {
            const Tag& tag = this->mergedInputTag();
            if (verbose_console) {
                print_tag(tag, std::format("{}::processOne(...)\t received tag at {:6}", this->name, _nSamplesProduced));
            }
            if (log_tags) {
                const auto& newTag = _tags.emplace_back(_nSamplesProduced, tag.map);
                if (_tagCallback) {
                    _tagCallback(newTag);
                }
            }
        }
        if (log_samples) {
            _samples.push_back(input);
        }
        _nSamplesProduced++;
        return input;
    }

    constexpr work::Status processBulk(std::span<const T> input, std::span<T> output) noexcept
    requires(UseProcessVariant == ProcessFunction::USE_PROCESS_BULK)
    {
        if (this->inputTagsPresent()) {
            const Tag& tag = this->mergedInputTag();
            if (verbose_console) {
                print_tag(tag, std::format("{}::processBulk(...{}, ...{})\t received tag at {:6}", this->name, input.size(), output.size(), _nSamplesProduced));
            }
            if (log_tags) {
                const auto& newTag = _tags.emplace_back(_nSamplesProduced, tag.map);
                if (_tagCallback) {
                    _tagCallback(newTag);
                }
            }
        }

        if (log_samples) {
            _samples.insert(_samples.end(), input.begin(), input.end());
        }

        _nSamplesProduced += static_cast<gr::Size_t>(input.size());
        std::memcpy(output.data(), input.data(), input.size() * sizeof(T));

        return work::Status::OK;
    }
};

GR_REGISTER_BLOCK("gr::testing::TagSink", gr::testing::TagSink, ([T], gr::testing::ProcessFunction::USE_PROCESS_ONE), [ float, double ])

template<typename T, ProcessFunction UseProcessVariant>
struct TagSink : public Block<TagSink<T, UseProcessVariant>> {
    using ClockSourceType = std::chrono::system_clock;
    PortIn<T> in;

    // settings
    gr::Size_t  n_samples_expected{0};
    float       sample_rate = 1000.0f;
    std::string signal_name;
    bool        log_tags        = true;
    bool        log_samples     = true;
    bool        verbose_console = false;

    GR_MAKE_REFLECTABLE(TagSink, in, n_samples_expected, sample_rate, signal_name, log_tags, log_samples, verbose_console);

    std::vector<T>   _samples{};
    std::vector<Tag> _tags{};
    gr::Size_t       _nSamplesProduced{0}; // for infinite samples the counter wraps around back to 0

    std::function<void(const Tag&)> _tagCallback{};

    void start() {
        if (verbose_console) {
            std::println("started sink {} aka. '{}'", this->unique_name, this->name);
        }
        _samples.clear();
        if (log_samples) {
            _samples.reserve(std::max(0UZ, static_cast<std::size_t>(n_samples_expected)));
        }
        _tags.clear();
    }

    void stop() {
        if (verbose_console) {
            std::println("stopped sink {} aka. '{}'", this->unique_name, this->name);
        }
    }

    constexpr void processOne(const T& input) // N.B. non-SIMD since we need a sample-by-sample accurate tag detection
    requires(UseProcessVariant == ProcessFunction::USE_PROCESS_ONE)
    {
        if (this->inputTagsPresent()) {
            const Tag& tag = this->mergedInputTag();
            if (verbose_console) {
                print_tag(tag, std::format("{}::processOne(...1)    \t received tag at {:6}", this->name, _nSamplesProduced));
            }
            if (log_tags) {
                const auto& newTag = _tags.emplace_back(_nSamplesProduced, tag.map);
                if (_tagCallback) {
                    _tagCallback(newTag);
                }
            }
        }
        if (log_samples) {
            _samples.push_back(input);
        }
        _nSamplesProduced++;
        if (n_samples_expected > 0 && _nSamplesProduced >= n_samples_expected) {
            this->requestStop();
        }
    }

    // template<gr::meta::t_or_simd<T> V>
    constexpr work::Status processBulk(std::span<const T> input)
    requires(UseProcessVariant == ProcessFunction::USE_PROCESS_BULK)
    {
        if (this->inputTagsPresent()) {
            const Tag& tag = this->mergedInputTag();
            if (verbose_console) {
                print_tag(tag, std::format("{}::processBulk(...{})\t received tag at {:6}", this->name, input.size(), _nSamplesProduced));
            }
            if (log_tags) {
                const auto& newTag = _tags.emplace_back(_nSamplesProduced, tag.map);
                if (_tagCallback) {
                    _tagCallback(newTag);
                }
            }
        }
        if (log_samples) {
            _samples.insert(_samples.end(), input.begin(), input.end());
        }
        _nSamplesProduced += static_cast<gr::Size_t>(input.size());
        return n_samples_expected > 0 && _nSamplesProduced >= n_samples_expected ? work::Status::DONE : work::Status::OK;
    }
};

} // namespace gr::testing

#endif // GNURADIO_TAGMONITORS_HPP
