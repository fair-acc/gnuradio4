#ifndef GNURADIO_SETTINGS_HPP
#define GNURADIO_SETTINGS_HPP

#include <atomic>
#include <chrono>
#include <concepts>
#include <mutex>
#include <optional>
#include <set>
#include <variant>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <gnuradio-4.0/BlockTraits.hpp>
#include <gnuradio-4.0/Tag.hpp>
#include <gnuradio-4.0/annotated.hpp>
#include <gnuradio-4.0/meta/formatter.hpp>
#include <gnuradio-4.0/reflection.hpp>

namespace gr {

namespace settings {

template<typename T>
inline constexpr static bool isSupportedVectorType() {
    if constexpr (gr::meta::vector_type<T>) {
        return std::is_arithmetic_v<typename T::value_type> || std::is_same_v<typename T::value_type, std::string>;
    } else {
        return false;
    }
}

template<typename T>
inline constexpr static bool isSupportedType() {
    return std::is_arithmetic_v<T> || std::is_same_v<T, std::string> || isSupportedVectorType<T>() || std::is_same_v<T, property_map>;
}

template<typename T, typename TMember>
inline constexpr static bool isWritableMember(TMember member) {
    return traits::port::is_not_any_port_or_collection<T> && !std::is_const_v<T> && is_writable(member) && settings::isSupportedType<T>();
}

template<typename T, typename TMember>
inline constexpr static bool isReadableMember(TMember member) {
    return traits::port::is_not_any_port_or_collection<T> && is_readable(member) && settings::isSupportedType<T>();
}

inline constexpr uint64_t convertTimePointToUint64Ns(const std::chrono::time_point<std::chrono::system_clock>& tp) {
    const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(tp.time_since_epoch()).count();
    return static_cast<uint64_t>(ns);
}

// pmtv::pmt comparison is needed to use it as a key of std::map
struct PMTCompare {
    bool operator()(const pmtv::pmt& lhs, const pmtv::pmt& rhs) const {
        // If the types are different, cast rhs to the type of lhs and compare
        if (lhs.index() != rhs.index()) {
            // TODO: throw if types are not the same?
            return lhs.index() < rhs.index();
        } else {
            if (std::holds_alternative<std::string>(lhs)) {
                return std::get<std::string>(lhs) < std::get<std::string>(rhs);
            } else if (std::holds_alternative<int>(lhs)) {
                return std::get<int>(lhs) < std::get<int>(rhs);
            } else {
                throw gr::exception("Invalid CtxSettings context type " + std::string(typeid(lhs).name()));
            }
        }
    }
};
} // namespace settings

struct ApplyStagedParametersResult {
    property_map forwardParameters; // parameters that should be forwarded to dependent child blocks
    property_map appliedParameters;
};

namespace detail {
template<class T>
inline constexpr void hash_combine(std::size_t& seed, const T& v) noexcept {
    std::hash<T> hasher;
    seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}
} // namespace detail

struct SettingsCtx {
    uint64_t  time    = 0ULL; // UTC-based time-stamp in ns, time from which the setting is valid, 0U is undefined time
    pmtv::pmt context = "";   // user-defined multiplexing context for which the setting is valid

    bool operator==(const SettingsCtx&) const = default;

    bool operator<(const SettingsCtx& other) {
        // order by time
        return time < other.time;
    }

    [[nodiscard]] std::size_t hash() const noexcept {
        std::size_t seed = 0;
        if (time != 0ULL) {
            detail::hash_combine(seed, time);
        }
        detail::hash_combine(seed, pmtv::to_base64(context));
        return seed;
    }
};

/**
 * @brief a concept verifying whether a processing block optionally provides a `settingsChanged` callback to react to
 * block configuration changes and/or to influence forwarded downstream parameters.
 *
 * Implementers may have:
 * 1. `settingsChanged(oldSettings, newSettings)`
 * 2. `settingsChanged(oldSettings, newSettings, forwardSettings)`
 *    - where `forwardSettings` is for influencing subsequent blocks. E.g., a decimating block might adjust the `sample_rate` for downstream blocks.
 */
template<typename BlockType>
concept HasSettingsChangedCallback = requires(BlockType* block, const property_map& oldSettings, property_map& newSettings) {
    { block->settingsChanged(oldSettings, newSettings) };
} or requires(BlockType* block, const property_map& oldSettings, property_map& newSettings, property_map& forwardSettings) {
    { block->settingsChanged(oldSettings, newSettings, forwardSettings) };
};

/**
 * @brief a concept verifying whether a processing block optionally provides a `reset` callback to react to
 * block reset requests (being called after the settings have been reverted(.
 */
template<typename TBlock>
concept HasSettingsResetCallback = requires(TBlock* block) {
    { block->reset() };
};

template<typename T>
concept SettingsLike = requires(T t, std::span<const std::string> parameter_keys, const std::string& parameter_key, const property_map& parameters, SettingsCtx ctx, const Tag& tag) {
    /**
     * @brief returns if there are stages settings that haven't been applied yet.
     */
    { t.changed() } -> std::same_as<bool>;

    /**
     * @brief stages new key-value pairs that shall replace the block field-based settings.
     * N.B. settings become only active after executing 'applyStagedParameters()' (usually done early on in the 'Block::work()' function)
     * @return key-value pairs that could not be set
     */
    { t.set(parameters, ctx) } -> std::same_as<property_map>;
    { t.set(parameters) } -> std::same_as<property_map>;

    /**
     * @brief updates parameters based on block input tags for those with keys stored in `autoUpdateParameters()`
     * Parameter changes to down-stream blocks is controlled via `autoForwardParameters()`
     */
    { t.autoUpdate(tag) } -> std::same_as<void>;

    /**
     * @brief return all available active block settings as key-value pairs
     */
    { t.get() } -> std::same_as<property_map>;

    /**
     * @brief return all available active block settings as key-value pairs for multiple keys
     */
    { t.get(parameter_keys) } -> std::same_as<property_map>;

    /**
     * @brief return available active block setting as key-value pair for a single key
     */
    { t.get(parameter_key) } -> std::same_as<std::optional<pmtv::pmt>>;

    /**
     * @brief return all available stored block settings for default SettingsCtx as key-value pairs
     */
    { t.getStored() } -> std::same_as<property_map>;

    /**
     * @brief return all available stored block settings as key-value pairs for multiple keys
     */
    { t.getStored(parameter_keys) } -> std::same_as<property_map>;
    { t.getStored(parameter_keys, ctx) } -> std::same_as<property_map>;

    /**
     * @brief return available stored block setting as key-value pair for a single key
     */
    { t.getStored(parameter_key) } -> std::same_as<std::optional<pmtv::pmt>>;
    { t.getStored(parameter_key, ctx) } -> std::same_as<std::optional<pmtv::pmt>>;

    /**
     * @brief return number of all stored parameters
     */
    { t.getNStoredParameters() } -> std::same_as<gr::Size_t>;

    /**
     * @brief return _storedParameters
     */
    { t.getStoredAll() } -> std::same_as<std::map<pmtv::pmt, std::vector<std::pair<SettingsCtx, property_map>>, settings::PMTCompare>>;

    /**
     * @brief returns the staged/not-yet-applied new parameters
     */
    { t.stagedParameters() } -> std::same_as<const property_map>;

    /**
     * @brief synchronise map-based with actual block field-based settings
     */
    { t.applyStagedParameters() } -> std::same_as<ApplyStagedParametersResult>;
    { t.applyStagedParameters(0ULL) } -> std::same_as<ApplyStagedParametersResult>;

    /**
     * @brief synchronises the map-based with the block's field-based parameters
     * (N.B. usually called after the staged parameters have been synchronised)
     */
    { t.updateActiveParameters() } -> std::same_as<void>;
};

struct SettingsBase {
    std::atomic_bool _changed{false};

    virtual ~SettingsBase() = default;

    void swap(SettingsBase& other) noexcept {
        if (this == &other) {
            return;
        }
        bool temp = _changed;
        // avoid CAS-loop since this called only during initialisation where there is no concurrent access possible.
        std::atomic_store_explicit(&_changed, std::atomic_load_explicit(&other._changed, std::memory_order_acquire), std::memory_order_release);
        other._changed = temp;
    }

    /**
     * @brief returns if there are stages settings that haven't been applied yet.
     */
    [[nodiscard]] bool changed() const noexcept { return _changed; }

    /**
     * @brief stages new key-value pairs that shall replace the block field-based settings.
     * N.B. settings become only active after executing 'applyStagedParameters()' (usually done early on in the 'Block::work()' function)
     * @return key-value pairs that could not be set
     */
    [[nodiscard]] virtual property_map set(const property_map& parameters, SettingsCtx ctx = {}) = 0;

    virtual void storeDefaults() = 0;
    virtual void resetDefaults() = 0;

    /**
     * @brief updates parameters based on block input tags for those with keys stored in `autoUpdateParameters()`
     * Parameter changes to down-stream blocks is controlled via `autoForwardParameters()`
     */
    virtual void autoUpdate(const Tag& tag) = 0;

    /**
     * @brief return all (or for selected multiple keys) available active block settings as key-value pairs
     */
    [[nodiscard]] virtual property_map get(std::span<const std::string> parameter_keys = {}) const noexcept = 0;

    /**
     * @brief return available active block setting as key-value pair for a single key
     */
    [[nodiscard]] virtual std::optional<pmtv::pmt> get(const std::string& parameter_key) const noexcept = 0;

    /**
     * @brief return all (or for selected multiple keys) stored block settings for provided context as key-value pairs
     */
    [[nodiscard]] virtual property_map getStored(std::span<const std::string> parameterKeys = {}, SettingsCtx ctx = {}) const noexcept = 0;

    /**
     * @brief return available stored block setting for provided context as key-value pair for a single key
     */
    [[nodiscard]] virtual std::optional<pmtv::pmt> getStored(const std::string& parameter_key, SettingsCtx ctx = {}) const noexcept = 0;

    /**
     * @brief return number of all stored parameters
     */
    [[nodiscard]] virtual gr::Size_t getNStoredParameters() const noexcept = 0;

    /**
     * @brief return _storedParameters
     */
    [[nodiscard]] virtual std::map<pmtv::pmt, std::vector<std::pair<SettingsCtx, property_map>>, settings::PMTCompare> getStoredAll() const noexcept = 0;

    /**
     * @brief returns the staged/not-yet-applied new parameters
     */
    [[nodiscard]] virtual const property_map stagedParameters(SettingsCtx ctx = {}) const = 0;

    [[nodiscard]] virtual std::set<std::string, std::less<>> autoUpdateParameters(SettingsCtx ctx = {}) noexcept = 0;

    [[nodiscard]] virtual std::set<std::string, std::less<>>& autoForwardParameters() noexcept = 0;

    /**
     * @brief synchronise map-based with actual block field-based settings
     * returns map with key-value tags that should be forwarded
     * to dependent/child blocks.
     */
    [[nodiscard]] virtual ApplyStagedParametersResult applyStagedParameters(std::uint64_t currentTime = 0ULL) = 0;

    /**
     * @brief synchronises the map-based with the block's field-based parameters
     * (N.B. usually called after the staged parameters have been synchronised)
     */
    virtual void updateActiveParameters() noexcept = 0;
};

namespace detail {
template<typename T>
concept HasBaseType = requires { typename std::remove_cvref_t<T>::base_t; };
}; // namespace detail

template<typename TBlock>
class BasicSettings : public SettingsBase {
    TBlock*                            _block = nullptr;
    mutable std::mutex                 _lock{};
    property_map                       _active{}; // copy of class field settings as pmt-style map
    property_map                       _staged{}; // parameters to become active before the next work() call
    std::set<std::string, std::less<>> _auto_update{};
    std::set<std::string, std::less<>> _auto_forward{};
    property_map                       _default_settings{};

public:
    BasicSettings()  = delete;
    ~BasicSettings() = default;

    explicit constexpr BasicSettings(TBlock& block) noexcept : SettingsBase(), _block(&block) {
        if constexpr (requires { &TBlock::settingsChanged; }) { // if settingsChanged is defined
            static_assert(HasSettingsChangedCallback<TBlock>, "if provided, settingsChanged must have either a `(const property_map& old, property_map& new, property_map& fwd)`"
                                                              "or `(const property_map& old, property_map& new)` parameter signatures.");
        }

        if constexpr (requires { &TBlock::reset; }) { // if reset is defined
            static_assert(HasSettingsResetCallback<TBlock>, "if provided, reset() may have no function parameters");
        }

        if constexpr (refl::is_reflectable<TBlock>()) {
            // register block-global description
            constexpr bool hasMetaInfo = requires(TBlock t) {
                {
                    unwrap_if_wrapped_t<decltype(t.meta_information)> {}
                } -> std::same_as<property_map>;
            };

            if constexpr (hasMetaInfo && requires(TBlock t) { t.description; }) {
                static_assert(std::is_same_v<std::remove_cvref_t<unwrap_if_wrapped_t<decltype(TBlock::description)>>, std::string_view>);
                _block->meta_information.value["description"] = std::string(_block->description);
            }

            // handle meta-information for UI and other non-processing-related purposes
            auto processOneMember = [this]<typename Member>(Member member) {
                using RawType         = std::remove_cvref_t<decltype(member(*_block))>;
                using Type            = unwrap_if_wrapped_t<RawType>;
                const auto memberName = std::string(get_display_name(member));

                if constexpr (hasMetaInfo && AnnotatedType<RawType>) {
                    _block->meta_information.value[memberName + "::description"]   = std::string(RawType::description());
                    _block->meta_information.value[memberName + "::documentation"] = std::string(RawType::documentation());
                    _block->meta_information.value[memberName + "::unit"]          = std::string(RawType::unit());
                    _block->meta_information.value[memberName + "::visible"]       = RawType::visible();
                }

                // detect whether field has one of the kDefaultTags signature
                if constexpr (settings::isWritableMember<Type>(member)) {
                    if constexpr (std::ranges::find(gr::tag::kDefaultTags, std::string_view(get_display_name_const(member).c_str())) != gr::tag::kDefaultTags.cend()) {
                        _auto_forward.emplace(memberName);
                    }
                    _auto_update.emplace(memberName);
                }
            };
            processMembers<TBlock>(processOneMember);
        }
    }

    constexpr BasicSettings(const BasicSettings& other) noexcept : SettingsBase(other) {
        BasicSettings temp(other);
        swap(temp);
    }

    constexpr BasicSettings(BasicSettings&& other) noexcept : SettingsBase(std::move(other)) {
        BasicSettings temp(std::move(other));
        swap(temp);
    }

    BasicSettings& operator=(const BasicSettings& other) noexcept {
        swap(other);
        return *this;
    }

    BasicSettings& operator=(BasicSettings&& other) noexcept {
        BasicSettings temp(std::move(other));
        swap(temp);
        return *this;
    }

    void swap(BasicSettings& other) noexcept {
        if (this == &other) {
            return;
        }
        SettingsBase::swap(other);
        std::swap(_block, other._block);
        std::scoped_lock lock(_lock, other._lock);
        std::swap(_active, other._active);
        std::swap(_staged, other._staged);
        std::swap(_auto_update, other._auto_update);
        std::swap(_auto_forward, other._auto_forward);
    }

    [[nodiscard]] property_map set(const property_map& parameters, SettingsCtx = {}) override {
        property_map ret;
        if constexpr (refl::is_reflectable<TBlock>()) {
            std::lock_guard lg(_lock);
            for (const auto& [key, value] : parameters) {
                bool isSet            = false;
                auto processOneMember = [&, this](auto member) {
                    using Type = unwrap_if_wrapped_t<std::remove_cvref_t<decltype(member(*_block))>>;
                    if constexpr (settings::isWritableMember<Type>(member)) {
                        const auto fieldName = std::string_view(get_display_name(member));
                        if (fieldName == key && std::holds_alternative<Type>(value)) {
                            if (_auto_update.contains(key)) {
                                _auto_update.erase(key);
                            }
                            _staged.insert_or_assign(key, value);
                            SettingsBase::_changed.store(true);
                            isSet = true;
                        }
                        if (fieldName == key && !std::holds_alternative<Type>(value)) {
                            throw std::invalid_argument([&key, &value] { // lazy evaluation
                                const std::size_t actual_index   = value.index();
                                const std::size_t required_index = meta::to_typelist<pmtv::pmt>::index_of<Type>(); // This too, as per your implementation.
                                return fmt::format("value for key '{}' has a wrong type. Index of actual type: {} ({}), Index of expected type: {} ({})", key, actual_index, "<missing pmt type>", required_index, gr::meta::type_name<Type>());
                            }());
                        }
                    }
                };
                processMembers<TBlock>(processOneMember);
                if (!isSet) {
                    ret.insert_or_assign(key, pmtv::pmt(value));
                }
            }
        }

        // copy items that could not be matched to the block's meta_information map (if available)
        if constexpr (requires(TBlock t) {
                          {
                              unwrap_if_wrapped_t<decltype(t.meta_information)> {}
                          } -> std::same_as<property_map>;
                      }) {
            updateMaps(ret, _block->meta_information);
        }

        return ret; // N.B. returns those <key:value> parameters that could not be set
    }

    void storeDefaults() override { this->storeDefaultSettings(_default_settings); }

    void resetDefaults() override {
        _staged     = _default_settings;
        std::ignore = applyStagedParameters();
        if constexpr (HasSettingsResetCallback<TBlock>) {
            _block->reset();
        }
    }

    void autoUpdate(const Tag& tag) override {
        const auto& parameters = tag.map;
        if constexpr (refl::is_reflectable<TBlock>()) {
            for (const auto& [key, value] : parameters) {
                auto processOneMember = [&](auto member) {
                    using Type = unwrap_if_wrapped_t<std::remove_cvref_t<decltype(member(*_block))>>;
                    if constexpr (settings::isWritableMember<Type>(member)) {
                        if (_auto_update.contains(key) && std::string(get_display_name(member)) == key && std::holds_alternative<Type>(value)) {
                            _staged.insert_or_assign(key, value);
                            SettingsBase::_changed.store(true);
                        }
                    }
                };
                processMembers<TBlock>(processOneMember);
            }
        }
    }

    [[nodiscard]] const property_map stagedParameters(SettingsCtx = {}) const noexcept override {
        std::lock_guard lg(_lock);
        return _staged;
    }

    [[nodiscard]] property_map get(std::span<const std::string> parameter_keys = {}) const noexcept override {
        std::lock_guard lg(_lock);
        if (parameter_keys.empty()) {
            return _active;
        }
        property_map ret;
        for (const auto& key : parameter_keys) {
            if (_active.contains(key)) {
                ret.insert_or_assign(key, _active.at(key));
            }
        }
        return ret;
    }

    [[nodiscard]] std::optional<pmtv::pmt> get(const std::string& parameter_key) const noexcept override {
        if constexpr (refl::is_reflectable<TBlock>()) {
            std::lock_guard lg(_lock);

            if (_active.contains(parameter_key)) {
                return {_active.at(parameter_key)};
            }
        }

        return std::nullopt;
    }

    [[nodiscard]] property_map getStored(std::span<const std::string> parameterKeys = {}, SettingsCtx = {}) const noexcept override {
        std::lock_guard lg(_lock);
        if (parameterKeys.empty()) {
            return _staged;
        }
        property_map ret;
        for (const auto& key : parameterKeys) {
            if (_staged.contains(key)) {
                ret.insert_or_assign(key, _staged.at(key));
            }
        }
        return ret;
    }

    [[nodiscard]] std::optional<pmtv::pmt> getStored(const std::string& parameter_key, SettingsCtx = {}) const noexcept override {
        auto res = getStored(std::array<std::string, 1>({parameter_key}));
        if (res.contains(parameter_key)) {
            return res.at(parameter_key);
        } else {
            return std::nullopt;
        }
    }

    [[nodiscard]] gr::Size_t getNStoredParameters() const noexcept override { return 1; } // Implemented only for compatibility

    [[nodiscard]] std::map<pmtv::pmt, std::vector<std::pair<SettingsCtx, property_map>>, settings::PMTCompare> getStoredAll() const noexcept override { return {}; } // Implemented only for compatibility

    [[nodiscard]] std::set<std::string, std::less<>> autoUpdateParameters(SettingsCtx = {}) noexcept override { return _auto_update; }

    [[nodiscard]] std::set<std::string, std::less<>>& autoForwardParameters() noexcept override { return _auto_forward; }

    /**
     * @brief synchronise map-based with actual block field-based settings
     * returns a structure containing three maps:
     *  - forwardParameters -- map with key-value tags that should be forwarded
     *    to dependent/child blocks.
     *  - appliedParameters -- map with peoperties that were successfully set
     */
    [[nodiscard]] ApplyStagedParametersResult applyStagedParameters(std::uint64_t = 0ULL) override {
        ApplyStagedParametersResult result;
        if constexpr (refl::is_reflectable<TBlock>()) {
            std::lock_guard lg(_lock);

            // prepare old settings if required
            property_map oldSettings;
            if constexpr (HasSettingsChangedCallback<TBlock>) {
                storeDefaultSettings(oldSettings);
            }

            // check if reset of settings should be performed
            if (_staged.contains(gr::tag::RESET_DEFAULTS)) {
                _staged.clear();
                resetDefaults();
            }

            // update staged and forward parameters based on member properties
            property_map staged;
            for (const auto& [key, stagedValue] : _staged) {
                auto applyOneMemberChanges = [&key, &staged, &result, &stagedValue, this](auto member) {
                    using RawType = std::remove_cvref_t<decltype(member(*_block))>;
                    using Type    = unwrap_if_wrapped_t<RawType>;
                    if constexpr (settings::isWritableMember<Type>(member)) {
                        if (std::string(get_display_name(member)) == key && std::holds_alternative<Type>(stagedValue)) {
                            if constexpr (is_annotated<RawType>()) {
                                if (member(*_block).validate_and_set(std::get<Type>(stagedValue))) {
                                    result.appliedParameters.insert_or_assign(key, stagedValue);
                                    if constexpr (HasSettingsChangedCallback<TBlock>) {
                                        staged.insert_or_assign(key, stagedValue);
                                    } else {
                                        std::ignore = staged; // help clang to see why staged is not unused
                                    }
                                } else {
                                    // TODO: replace with pmt error message on msgOut port (to note: clang compiler bug/issue)
#if !defined(__EMSCRIPTEN__) && !defined(__clang__)
                                    fmt::print(stderr, " cannot set field {}({})::{} = {} to {} due to limit constraints [{}, {}] validate func is {} defined\n", //
                                        _block->unique_name, _block->name, member(*_block), std::get<Type>(stagedValue),                                          //
                                        std::string(get_display_name(member)), RawType::LimitType::MinRange,
                                        RawType::LimitType::MaxRange, //
                                        RawType::LimitType::ValidatorFunc == nullptr ? "not" : "");
#else
                                    fmt::print(stderr, " cannot set field {}({})::{} = {} to {} due to limit constraints [{}, {}] validate func is {} defined\n", //
                                        "_block->unique_name", "_block->name", "member(*_block)", std::get<Type>(stagedValue),                                    //
                                        std::string(get_display_name(member)), RawType::LimitType::MinRange,
                                        RawType::LimitType::MaxRange, //
                                        RawType::LimitType::ValidatorFunc == nullptr ? "not" : "");
#endif
                                }
                            } else {
                                member(*_block) = std::get<Type>(stagedValue);
                                result.appliedParameters.insert_or_assign(key, stagedValue);
                                if constexpr (HasSettingsChangedCallback<TBlock>) {
                                    staged.insert_or_assign(key, stagedValue);
                                } else {
                                    std::ignore = staged; // help clang to see why staged is not unused
                                }
                            }
                        }
                        if (_auto_forward.contains(key)) {
                            result.forwardParameters.insert_or_assign(key, stagedValue);
                        }
                    }
                };
                processMembers<TBlock>(applyOneMemberChanges);
            }

            // update active parameters
            auto update_active = [this](auto member) {
                using Type = unwrap_if_wrapped_t<std::remove_cvref_t<decltype(member(*_block))>>;
                if constexpr (settings::isReadableMember<Type>(member)) {
                    _active.insert_or_assign(get_display_name(member), static_cast<Type>(member(*_block)));
                }
            };
            processMembers<TBlock>(update_active);

            // invoke user-callback function if staged is not empty
            if (!staged.empty()) {
                if constexpr (requires { _block->settingsChanged(/* old settings */ _active, /* new settings */ staged); }) {
                    _block->settingsChanged(/* old settings */ oldSettings, /* new settings */ staged);
                } else if constexpr (requires { _block->settingsChanged(/* old settings */ _active, /* new settings */ staged, /* new forward settings */ result.forwardParameters); }) {
                    _block->settingsChanged(/* old settings */ oldSettings, /* new settings */ staged, /* new forward settings */ result.forwardParameters);
                }
            }

            if (_staged.contains(gr::tag::STORE_DEFAULTS)) {
                storeDefaults();
            }

            if constexpr (HasSettingsResetCallback<TBlock>) {
                if (_staged.contains(gr::tag::RESET_DEFAULTS)) {
                    _block->reset();
                }
            }

            _staged.clear();
        }

        SettingsBase::_changed.store(false);
        return result;
    }

    void updateActiveParameters() noexcept override {
        if constexpr (refl::is_reflectable<TBlock>()) {
            std::lock_guard lg(_lock);
            auto            processOneMember = [&, this](auto member) {
                using Type = unwrap_if_wrapped_t<std::remove_cvref_t<decltype(member(*_block))>>;
                if constexpr (settings::isReadableMember<Type>(member)) {
                    _active.insert_or_assign(get_display_name(member), static_cast<Type>(member(*_block)));
                }
            };
            processMembers<TBlock>(processOneMember);
        }
    }

private:
    void storeDefaultSettings(property_map& oldSettings) {
        // take a copy of the field -> map value of the old settings
        if constexpr (refl::is_reflectable<TBlock>()) {
            auto processOneMember = [&, this](auto member) {
                using Type = unwrap_if_wrapped_t<std::remove_cvref_t<decltype(member(*_block))>>;
                if constexpr (settings::isReadableMember<Type>(member)) {
                    oldSettings.insert_or_assign(get_display_name(member), pmtv::pmt(member(*_block)));
                }
            };
            processMembers<TBlock>(processOneMember);
        }
    }

    template<typename T, typename Func>
    inline constexpr static void processMembers(Func func) {
        if constexpr (detail::HasBaseType<T>) {
            refl::util::for_each(refl::reflect<typename std::remove_cvref_t<T>::base_t>().members, func);
        }
        refl::util::for_each(refl::reflect<T>().members, func);
    }
};

static_assert(SettingsLike<BasicSettings<int>>);

} // namespace gr

namespace std {
template<>
struct hash<gr::SettingsCtx> {
    [[nodiscard]] size_t operator()(const gr::SettingsCtx& ctx) const noexcept { return ctx.hash(); }
};
} // namespace std

#endif // GNURADIO_SETTINGS_HPP
