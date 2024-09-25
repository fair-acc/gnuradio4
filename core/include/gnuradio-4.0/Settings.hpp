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

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat-nonliteral"
#include <fmt/chrono.h>
#pragma GCC diagnostic pop

#if defined(__GNUC__) || defined(__clang__)
#define NO_INLINE __attribute__((noinline))
#else
#define NO_INLINE
#endif

namespace gr {

namespace detail {
template<typename T>
concept HasBaseType = requires { typename std::remove_cvref_t<T>::base_t; };
}; // namespace detail

namespace settings {

template<typename T>
inline constexpr static bool isSupportedVectorType() {
    if constexpr (gr::meta::vector_type<T>) {
        return std::is_arithmetic_v<typename T::value_type> || std::is_same_v<typename T::value_type, std::string> //
               || std::is_same_v<typename T::value_type, std::complex<double>> || std::is_same_v<typename T::value_type, std::complex<float>>;
    } else {
        return false;
    }
}

template<typename T>
inline constexpr static bool isSupportedType() {
    return std::is_arithmetic_v<T> || std::is_same_v<T, std::string> || isSupportedVectorType<T>() || std::is_same_v<T, property_map> //
           || std::is_same_v<T, std::complex<double>> || std::is_same_v<T, std::complex<float>>;
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

static auto nullMatchPred = [](auto, auto, auto) { return std::nullopt; };

inline std::strong_ordering comparePmt(const pmtv::pmt& lhs, const pmtv::pmt& rhs) {
    // If the types are different, cast rhs to the type of lhs and compare
    if (lhs.index() != rhs.index()) {
        // TODO: throw if types are not the same?
        return lhs.index() <=> rhs.index();
    } else {
        if (std::holds_alternative<std::string>(lhs)) {
            return std::get<std::string>(lhs) <=> std::get<std::string>(rhs);
        } else if (std::holds_alternative<int>(lhs)) {
            return std::get<int>(lhs) <=> std::get<int>(rhs);
        } else {
            throw gr::exception("Invalid CtxSettings context type " + std::string(typeid(lhs).name()));
        }
    }
}

// pmtv::pmt comparison is needed to use it as a key of std::map
struct PMTCompare {
    bool operator()(const pmtv::pmt& lhs, const pmtv::pmt& rhs) const { return comparePmt(lhs, rhs) == std::strong_ordering::less; }
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
    std::uint64_t time    = 0ULL; // UTC-based time-stamp in ns, time from which the setting is valid, 0U is undefined time
    pmtv::pmt     context = "";   // user-defined multiplexing context for which the setting is valid

    bool operator==(const SettingsCtx&) const = default;

    auto operator<=>(const SettingsCtx& other) const {
        // First compare time
        if (auto cmp = time <=> other.time; cmp != std::strong_ordering::equal) {
            return cmp;
        }
        // Then compare context
        return settings::comparePmt(context, other.context);
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

struct SettingsBase {
    virtual ~SettingsBase() = default;

    /**
     * @brief returns if there are stored settings that haven't been applied yet.
     */
    [[nodiscard]] virtual bool changed() const noexcept    = 0;
    virtual void               setChanged(bool b) noexcept = 0;

    /**
     * @brief Set initial parameters provided in the Block constructor to ensure they are available during Settingd::init()
     */
    virtual void setInitBlockParameters(const property_map& parameters) = 0;

    /**
     * @brief initialize settings, set init parameters provided in the Block constructor
     */
    virtual void init() = 0;

    /**
     * @brief Add new key-value pairs to stored parameters.
     * N.B. settings become staged after calling activateContext(), and after executing 'applyStagedParameters()' settings are applied (usually done early on in the 'Block::work()' function)
     * @return key-value pairs that could not be set
     */
    [[nodiscard]] virtual property_map set(const property_map& parameters, SettingsCtx ctx = {}) = 0;

    /**
     * @brief Add new key-value pairs to stagedParameters. The changes do not affect storedParameters.
     * @return key-value pairs that could not be set
     */
    [[nodiscard]] virtual property_map setStaged(const property_map& parameters) = 0;

    virtual void storeDefaults() = 0;
    virtual void resetDefaults() = 0;

    /**
     * @brief Set new activate context and set staged parameters
     * @return best match context or std::nullopt if best match context is not found in storage
     */
    [[nodiscard]] virtual std::optional<SettingsCtx> activateContext(SettingsCtx ctx = {}) = 0;

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
    [[nodiscard]] virtual std::optional<property_map> getStored(std::span<const std::string> parameterKeys = {}, SettingsCtx ctx = {}) const noexcept = 0;

    /**
     * @brief return available stored block setting for provided context as key-value pair for a single key
     */
    [[nodiscard]] virtual std::optional<pmtv::pmt> getStored(const std::string& parameter_key, SettingsCtx ctx = {}) const noexcept = 0;

    /**
     * @brief return number of all sets of stored parameters
     */
    [[nodiscard]] virtual gr::Size_t getNStoredParameters() const noexcept = 0;

    /**
     * @brief return number of sets of auto update parameters
     */
    [[nodiscard]] virtual gr::Size_t getNAutoUpdateParameters() const noexcept = 0;

    /**
     * @brief return _storedParameters
     */
    [[nodiscard]] virtual std::map<pmtv::pmt, std::vector<std::pair<SettingsCtx, property_map>>, settings::PMTCompare> getStoredAll() const noexcept = 0;

    /**
     * @brief returns the staged/not-yet-applied new parameters
     */
    [[nodiscard]] virtual const property_map& stagedParameters() const = 0;

    [[nodiscard]] virtual std::set<std::string> autoUpdateParameters(SettingsCtx ctx = {}) noexcept = 0;

    [[nodiscard]] virtual std::set<std::string>& autoForwardParameters() noexcept = 0;

    [[nodiscard]] virtual const property_map& defaultParameters() const noexcept = 0;

    [[nodiscard]] virtual const property_map& activeParameters() const noexcept = 0;

    /**
     * @brief synchronise map-based with actual block field-based settings
     * returns map with key-value tags that should be forwarded
     * to dependent/child blocks.
     */
    [[nodiscard]] virtual ApplyStagedParametersResult applyStagedParameters() = 0;

    /**
     * @brief synchronises the map-based with the block's field-based parameters
     * (N.B. usually called after the staged parameters have been synchronised)
     */
    virtual void updateActiveParameters() noexcept = 0;
}; // struct SettingsBase

template<typename TBlock>
class CtxSettings : public SettingsBase {
    /**
     * A predicate for matching two contexts
     * The third "attempt" parameter indicates the current round of matching being done.
     * This is useful for hierarchical matching schemes,
     * e.g. in the first round the predicate could look for almost exact matches only,
     * then in a a second round (attempt=1) it could be more forgiving, given that there are no exact matches available.
     *
     * The predicate will be called until it returns "true" (a match is found), or until it returns std::nullopt,
     * which indicates that no matches were found and there is no chance of matching anything in a further round.
     */
    using MatchPredicate = std::function<std::optional<bool>(const pmtv::pmt&, const pmtv::pmt&, std::size_t)>;

    TBlock*            _block = nullptr;
    std::atomic_bool   _changed{false};
    mutable std::mutex _mutex{};

    // key: SettingsCtx.context, value: queue of parameters with the same SettingsCtx.context but for different time
    mutable std::map<pmtv::pmt, std::vector<std::pair<SettingsCtx, property_map>>, settings::PMTCompare> _storedParameters{};
    property_map                                                                                         _defaultParameters{};
    // Store the initial parameters provided in the Block constructor. These parameters cannot be set directly in the constructor
    // because `_defaultParameters` cannot be initialized using Settings::storeDefaults() within the Block constructor.
    // Instead, we store them now and set them later in the Block::init method.
    property_map                                 _initBlockParameters{};
    std::set<std::string>                        _allWritableMembers{};   // all `isWritableMember` class members
    std::map<SettingsCtx, std::set<std::string>> _autoUpdateParameters{}; // for each SettingsCtx auto updated members are stored separately
    std::set<std::string>                        _autoForwardParameters{};
    MatchPredicate                               _matchPred = settings::nullMatchPred;
    SettingsCtx                                  _activeCtx{};
    property_map                                 _stagedParameters{};
    property_map                                 _activeParameters{};

    const std::size_t _timePrecisionTolerance = 100; // ns, now used for emscripten

    template<typename Func>
    inline constexpr static void processMembers(Func func) {
        // reflect and iterate via known base-class member fields
        refl::util::for_each(refl::reflect<typename std::remove_cvref_t<TBlock>::base_t>().members, func);
        // reflect and iterate via known derived class member fields
        refl::util::for_each(refl::reflect<TBlock>().members, func);
    }

public:
    // Settings configuration
    std::uint64_t expiry_time{std::numeric_limits<std::uint64_t>::max()}; // in ns, expiry time of parameter set after the last use, std::numeric_limits<std::uint64_t>::max() == no expiry time

public:
    explicit CtxSettings(TBlock& block, MatchPredicate matchPred = settings::nullMatchPred) noexcept : SettingsBase(), _block(&block), _matchPred(matchPred) {
        if constexpr (requires { &TBlock::settingsChanged; }) { // if settingsChanged is defined
            static_assert(HasSettingsChangedCallback<TBlock>, "if provided, settingsChanged must have either a `(const property_map& old, property_map& new, property_map& fwd)`"
                                                              "or `(const property_map& old, property_map& new)` paremeter signatures.");
        }

        if constexpr (requires { &TBlock::reset; }) { // if reset is defined
            static_assert(HasSettingsResetCallback<TBlock>, "if provided, reset() may have no function parameters");
        }

        if constexpr (refl::is_reflectable<TBlock>()) {
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
            auto processOneMember = [&]<typename TFieldMeta>(TFieldMeta member) {
                using RawType         = std::remove_cvref_t<typename TFieldMeta::value_type>;
                using Type            = unwrap_if_wrapped_t<RawType>;
                const auto memberName = std::string(get_display_name(member));

                if constexpr (hasMetaInfo && AnnotatedType<RawType>) {
                    _block->meta_information.value[memberName + "::description"]   = std::string(RawType::description());
                    _block->meta_information.value[memberName + "::documentation"] = std::string(RawType::documentation());
                    _block->meta_information.value[memberName + "::unit"]          = std::string(RawType::unit());
                    _block->meta_information.value[memberName + "::visible"]       = RawType::visible();
                }

                // detect whether field has one of the DEFAULT_TAGS signature
                if constexpr (settings::isWritableMember<Type>(member)) {
                    if constexpr (std::ranges::find(gr::tag::kDefaultTags, std::string_view(get_display_name_const(member).c_str())) != gr::tag::kDefaultTags.cend()) {
                        _autoForwardParameters.emplace(memberName);
                    }
                    _allWritableMembers.emplace(memberName);
                }
            };
            processMembers(processOneMember);
        }
    }

    CtxSettings(const CtxSettings& other) {
        std::scoped_lock lock(_mutex, other._mutex);
        copyFrom(other);
    }

    CtxSettings(CtxSettings&& other) noexcept {
        std::scoped_lock lock(_mutex, other._mutex);
        moveFrom(other);
    }

    CtxSettings& operator=(const CtxSettings& other) noexcept {
        if (this == &other) {
            return *this;
        }

        std::scoped_lock lock(_mutex, other._mutex);
        copyFrom(other);
        return *this;
    }

    CtxSettings& operator=(CtxSettings&& other) noexcept {
        if (this == &other) {
            return *this;
        }
        std::scoped_lock lock(_mutex, other._mutex);
        moveFrom(other);
        return *this;
    }

private:
    void copyFrom(const CtxSettings& other) {
        _block = other._block;
        std::atomic_store_explicit(&_changed, std::atomic_load_explicit(&other._changed, std::memory_order_acquire), std::memory_order_release);
        _activeParameters      = other._activeParameters;
        _storedParameters      = other._storedParameters;
        _defaultParameters     = other._defaultParameters;
        _allWritableMembers    = other._allWritableMembers;
        _autoUpdateParameters  = other._autoUpdateParameters;
        _autoForwardParameters = other._autoForwardParameters;
        _matchPred             = other._matchPred;
        _activeCtx             = other._activeCtx;
    }

    void moveFrom(CtxSettings& other) noexcept {
        _block = std::exchange(other._block, nullptr);
        std::atomic_store_explicit(&_changed, std::atomic_load_explicit(&other._changed, std::memory_order_acquire), std::memory_order_release);
        _activeParameters      = std::move(other._activeParameters);
        _storedParameters      = std::move(other._storedParameters);
        _defaultParameters     = std::move(other._defaultParameters);
        _allWritableMembers    = std::move(other._allWritableMembers);
        _autoUpdateParameters  = std::move(other._autoUpdateParameters);
        _autoForwardParameters = std::move(other._autoForwardParameters);
        _matchPred             = std::exchange(other._matchPred, settings::nullMatchPred);
        _activeCtx             = std::exchange(other._activeCtx, {});
    }

public:
    [[nodiscard]] bool changed() const noexcept override { return _changed; }

    void setChanged(bool b) noexcept override { _changed.store(b); }

    void setInitBlockParameters(const property_map& parameters) override { _initBlockParameters = parameters; }

    NO_INLINE void init() override {
        storeDefaults();

        if (const property_map failed = set(_initBlockParameters); !failed.empty()) {
            throw gr::exception(fmt::format("settings could not be applied: {}", failed));
        }

        if (const auto failed = activateContext(); failed == std::nullopt) {
            throw gr::exception("Settings for context could not be activated");
        }
    }

    [[nodiscard]] property_map set(const property_map& parameters, SettingsCtx ctx = {}) override {
        property_map ret;
        if constexpr (refl::is_reflectable<TBlock>()) {
            std::lock_guard lg(_mutex);
            if (ctx.time == 0ULL) {
                ctx.time = settings::convertTimePointToUint64Ns(std::chrono::system_clock::now());
            }
#ifdef __EMSCRIPTEN__
            resolveDuplicateTimestamp(ctx);
#endif
            // initialize with empty property_map when best match parameters not found
            property_map newParameters = getBestMatchStoredParameters(ctx).value_or(_defaultParameters);
            if (!_autoUpdateParameters.contains(ctx)) {
                _autoUpdateParameters[ctx] = getBestMatchAutoUpdateParameters(ctx).value_or(_allWritableMembers);
            }
            auto& currentAutoUpdateParameters = _autoUpdateParameters[ctx];

            for (const auto& [key, value] : parameters) {
                bool isSet            = false;
                auto processOneMember = [&]<typename TFieldMeta>(TFieldMeta member) {
                    using Type = unwrap_if_wrapped_t<std::remove_cvref_t<typename TFieldMeta::value_type>>;
                    if constexpr (settings::isWritableMember<Type>(member)) {
                        const auto fieldName = std::string_view(get_display_name(member));
                        if (fieldName == key && std::holds_alternative<Type>(value)) {
                            if (currentAutoUpdateParameters.contains(key)) {
                                currentAutoUpdateParameters.erase(key);
                            }
                            newParameters.insert_or_assign(key, value);
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
                processMembers(processOneMember);
                if (!isSet) {
                    ret.insert_or_assign(key, pmtv::pmt(value));
                }
            }
            addStoredParameters(newParameters, ctx);
            removeExpiredStoredParameters();
        }

        // copy items that could not be matched to the node's meta_information map (if available)
        if constexpr (requires(TBlock t) {
                          {
                              unwrap_if_wrapped_t<decltype(t.meta_information)> {}
                          } -> std::same_as<property_map>;
                      }) {
            updateMaps(ret, _block->meta_information);
        }

        return ret; // N.B. returns those <key:value> parameters that could not be set
    }

    [[nodiscard]] property_map setStaged(const property_map& parameters) override {
        std::lock_guard lg(_mutex);
        return setStagedImpl(parameters);
    }

    void storeDefaults() override { this->storeCurrentParameters(_defaultParameters); }

    NO_INLINE void resetDefaults() override {
        // add default parameters to stored and apply the parameters
        auto ctx = SettingsCtx{settings::convertTimePointToUint64Ns(std::chrono::system_clock::now()), ""};
#ifdef __EMSCRIPTEN__
        resolveDuplicateTimestamp(ctx);
#endif
        addStoredParameters(_defaultParameters, ctx);
        std::ignore = activateContext();
        std::ignore = applyStagedParameters();

        removeExpiredStoredParameters();

        if constexpr (HasSettingsResetCallback<TBlock>) {
            _block->reset();
        }
    }

    [[nodiscard]] NO_INLINE std::optional<SettingsCtx> activateContext(SettingsCtx ctx = {}) override {
        if (ctx.time == 0ULL) {
            ctx.time = settings::convertTimePointToUint64Ns(std::chrono::system_clock::now());
#ifdef __EMSCRIPTEN__
            ctx.time += _timePrecisionTolerance;
#endif
        }

        const std::optional<SettingsCtx> bestMatchSettingsCtx = findBestMatchSettingsCtx(ctx);
        if (!bestMatchSettingsCtx || bestMatchSettingsCtx == _activeCtx) {
            return bestMatchSettingsCtx;
        }

        if (bestMatchSettingsCtx.value().context == _activeCtx.context) {
            std::optional<property_map> parameters = getBestMatchStoredParameters(ctx);
            if (parameters) {
                const std::set<std::string>& currentAutoUpdateParams = _autoUpdateParameters.at(bestMatchSettingsCtx.value());
                // auto                         notAutoUpdateView       = parameters.value() | std::views::filter([&](const auto& pair) { return !currentAutoUpdateParams.contains(pair.first); });
                // property_map                 notAutoUpdateParams(notAutoUpdateView.begin(), notAutoUpdateView.end());

                // the following is more compile-time friendly
                property_map notAutoUpdateParams;
                for (const auto& pair : parameters.value()) {
                    if (!currentAutoUpdateParams.contains(pair.first)) {
                        notAutoUpdateParams.insert(pair);
                    }
                }

                std::ignore = setStagedImpl(std::move(notAutoUpdateParams));
                _activeCtx  = bestMatchSettingsCtx.value();
                setChanged(true);
            }
        } else {
            std::optional<property_map> parameters = getBestMatchStoredParameters(ctx);
            if (parameters) {
                _stagedParameters.insert(parameters.value().begin(), parameters.value().end());
                _activeCtx = bestMatchSettingsCtx.value();
                setChanged(true);
            } else {
                return std::nullopt;
            }
        }

        return bestMatchSettingsCtx;
    }

    NO_INLINE void autoUpdate(const Tag& tag) override {
        if constexpr (refl::is_reflectable<TBlock>()) {
            std::lock_guard lg(_mutex);
            const auto      tagCtx = createSettingsCtxFromTag(tag);

            SettingsCtx ctx;
            if (tagCtx != std::nullopt) {
                const auto bestMatchSettingsCtx = activateContext(tagCtx.value());
                if (bestMatchSettingsCtx == std::nullopt) {
                    ctx = _activeCtx;
                } else {
                    ctx = bestMatchSettingsCtx.value();
                }
            } else {
                ctx = _activeCtx;
            }

            const bool activeCtxChanged = _activeCtx == ctx;

            const auto autoUpdateParameters = _autoUpdateParameters.find(ctx);
            if (autoUpdateParameters == _autoUpdateParameters.end()) {
                return;
            }

            const property_map& parameters = tag.map;
            bool                wasChanged = false;
            for (const auto& [key, value] : parameters) {
                auto processOneMember = [&]<typename TFieldMeta>(TFieldMeta member) {
                    using Type = unwrap_if_wrapped_t<std::remove_cvref_t<typename TFieldMeta::value_type>>;
                    if constexpr (settings::isWritableMember<Type>(member)) {
                        if (std::string_view(get_display_name(member)) == key && autoUpdateParameters->second.contains(key) && std::holds_alternative<Type>(value)) {
                            _stagedParameters.insert_or_assign(key, value);
                            wasChanged = true;
                        }
                    }
                };
                processMembers(processOneMember);
            }

            if (tagCtx == std::nullopt && !wasChanged) { // not context and no parameters in the Tag
                _stagedParameters.clear();
                setChanged(false);
            } else if (activeCtxChanged || wasChanged) {
                setChanged(true);
            }
        }
    }

    [[nodiscard]] property_map get(std::span<const std::string> parameterKeys = {}) const noexcept override {
        std::lock_guard lg(_mutex);
        if (parameterKeys.empty()) {
            return _activeParameters;
        }
        property_map ret;
        for (const auto& key : parameterKeys) {
            if (_activeParameters.contains(key)) {
                ret.insert_or_assign(key, _activeParameters.at(key));
            }
        }
        return ret;
    }

    [[nodiscard]] std::optional<pmtv::pmt> get(const std::string& parameterKey) const noexcept override {
        auto res = get(std::array<std::string, 1>({parameterKey}));
        if (res.contains(parameterKey)) {
            return res.at(parameterKey);
        } else {
            return std::nullopt;
        }
    }

    [[nodiscard]] NO_INLINE std::optional<property_map> getStored(std::span<const std::string> parameterKeys = {}, SettingsCtx ctx = {}) const noexcept override {
        std::lock_guard lg(_mutex);
        if (ctx.time == 0ULL) {
            ctx.time = settings::convertTimePointToUint64Ns(std::chrono::system_clock::now());
        }
#ifdef __EMSCRIPTEN__
        ctx.time += _timePrecisionTolerance;
#endif
        std::optional<property_map> allBestMatchParameters = this->getBestMatchStoredParameters(ctx);

        if (allBestMatchParameters == std::nullopt) {
            return std::nullopt;
        }

        if (parameterKeys.empty()) {
            return allBestMatchParameters;
        }
        property_map ret;
        for (const auto& key : parameterKeys) {
            if (allBestMatchParameters.value().contains(key)) {
                ret.insert_or_assign(key, allBestMatchParameters.value().at(key));
            }
        }
        return ret;
    }

    [[nodiscard]] std::optional<pmtv::pmt> getStored(const std::string& parameterKey, SettingsCtx ctx = {}) const noexcept override {
        auto res = getStored(std::array<std::string, 1>({parameterKey}), ctx);

        if (res != std::nullopt && res.value().contains(parameterKey)) {
            return res.value().at(parameterKey);
        } else {
            return std::nullopt;
        }
    }

    [[nodiscard]] gr::Size_t getNStoredParameters() const noexcept override {
        std::lock_guard lg(_mutex);
        gr::Size_t      nParameters{0};
        for (const auto& stored : _storedParameters) {
            nParameters += static_cast<gr::Size_t>(stored.second.size());
        }
        return nParameters;
    }

    [[nodiscard]] gr::Size_t getNAutoUpdateParameters() const noexcept override {
        std::lock_guard lg(_mutex);
        return static_cast<gr::Size_t>(_autoUpdateParameters.size());
    }

    [[nodiscard]] std::map<pmtv::pmt, std::vector<std::pair<SettingsCtx, property_map>>, settings::PMTCompare> getStoredAll() const noexcept override { return _storedParameters; }

    [[nodiscard]] const property_map& stagedParameters() const noexcept override {
        std::lock_guard lg(_mutex);
        return _stagedParameters;
    }

    [[nodiscard]] NO_INLINE std::set<std::string> autoUpdateParameters(SettingsCtx ctx = {}) noexcept override {
        auto bestMatchSettingsCtx = findBestMatchSettingsCtx(ctx);
        return bestMatchSettingsCtx == std::nullopt ? std::set<std::string>() : _autoUpdateParameters[bestMatchSettingsCtx.value()];
    }

    [[nodiscard]] NO_INLINE std::set<std::string>& autoForwardParameters() noexcept override { return _autoForwardParameters; }

    [[nodiscard]] NO_INLINE const property_map& defaultParameters() const noexcept override { return _defaultParameters; }

    [[nodiscard]] NO_INLINE const property_map& activeParameters() const noexcept override { return _activeParameters; }

    [[nodiscard]] NO_INLINE ApplyStagedParametersResult applyStagedParameters() override {
        ApplyStagedParametersResult result;
        if constexpr (refl::is_reflectable<TBlock>()) {
            std::lock_guard lg(_mutex);

            // prepare old settings if required
            property_map oldSettings;
            if constexpr (HasSettingsChangedCallback<TBlock>) {
                storeCurrentParameters(oldSettings);
            }

            // check if reset of settings should be performed
            if (_stagedParameters.contains(gr::tag::RESET_DEFAULTS)) {
                resetDefaults();
            }

            // update staged and forward parameters based on member properties
            property_map staged;
            for (const auto& [key, stagedValue] : _stagedParameters) {
                auto applyOneMemberChanges = [&key, &staged, &result, &stagedValue, this]<typename TFieldMeta>(TFieldMeta member) {
                    using RawType = std::remove_cvref_t<typename TFieldMeta::value_type>;
                    using Type    = unwrap_if_wrapped_t<RawType>;
                    if constexpr (settings::isWritableMember<Type>(member)) {
                        if (std::string(get_display_name(member)) == key && std::holds_alternative<Type>(stagedValue)) {
                            if constexpr (is_annotated<RawType>()) {
                                if (member(*_block).validate_and_set(std::get<Type>(stagedValue))) {
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
                                        "_block->uniqueName", "_block->name", member(*_block), std::get<Type>(stagedValue),                                       //
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
                        if (_autoForwardParameters.contains(key)) {
                            result.forwardParameters.insert_or_assign(key, stagedValue);
                        }
                    }
                };
                processMembers(applyOneMemberChanges);
            }

            updateActiveParametersImpl();

            // invoke user-callback function if staged is not empty
            if (!staged.empty()) {
                if constexpr (requires { _block->settingsChanged(/* old settings */ _activeParameters, /* new settings */ staged); }) {
                    _block->settingsChanged(/* old settings */ oldSettings, /* new settings */ staged);
                } else if constexpr (requires { _block->settingsChanged(/* old settings */ _activeParameters, /* new settings */ staged, /* new forward settings */ result.forwardParameters); }) {
                    _block->settingsChanged(/* old settings */ oldSettings, /* new settings */ staged, /* new forward settings */ result.forwardParameters);
                }
            }

            if (_stagedParameters.contains(gr::tag::STORE_DEFAULTS)) {
                storeDefaults();
            }

            if constexpr (HasSettingsResetCallback<TBlock>) {
                if (_stagedParameters.contains(gr::tag::RESET_DEFAULTS)) {
                    _block->reset();
                }
            }
        }
        _stagedParameters.clear();
        _changed.store(false);
        return result;
    }

    NO_INLINE void updateActiveParameters() noexcept override {
        if constexpr (refl::is_reflectable<TBlock>()) {
            std::lock_guard lg(_mutex);
            updateActiveParametersImpl();
        }
    }

private:
    NO_INLINE void updateActiveParametersImpl() noexcept {
        auto processOneMember = [&, this]<typename TFieldMeta>(TFieldMeta member) {
            using Type = unwrap_if_wrapped_t<std::remove_cvref_t<typename TFieldMeta::value_type>>;
            if constexpr (settings::isReadableMember<Type>(member)) {
                _activeParameters.insert_or_assign(get_display_name_const(member).str(), static_cast<Type>(member(*_block)));
            }
        };
        processMembers(processOneMember);
    }

    [[nodiscard]] NO_INLINE std::optional<pmtv::pmt> findBestMatchCtx(const pmtv::pmt& contextToSearch) const {
        if (_storedParameters.empty()) {
            return std::nullopt;
        }

        // exact match
        if (_storedParameters.find(contextToSearch) != _storedParameters.end()) {
            return contextToSearch;
        }

        // retry until we either get a match or std::nullopt
        for (std::size_t attempt = 0;; ++attempt) {
            for (const auto& i : _storedParameters) {
                const auto matches = _matchPred(i.first, contextToSearch, attempt);
                if (!matches) {
                    return std::nullopt;
                } else if (*matches) {
                    return i.first; // return the best matched SettingsCtx.context
                }
            }
        }
        return std::nullopt;
    }

    [[nodiscard]] NO_INLINE std::optional<SettingsCtx> findBestMatchSettingsCtx(const SettingsCtx& ctx) const {
        const auto bestMatchCtx = findBestMatchCtx(ctx.context);
        if (bestMatchCtx == std::nullopt) {
            return std::nullopt;
        }
        const auto& vec = _storedParameters[bestMatchCtx.value()];
        if (vec.empty()) {
            return std::nullopt;
        }
        if (ctx.time == 0ULL || vec.back().first.time <= ctx.time) {
            return vec.back().first;
        } else {
            auto lower = std::ranges::lower_bound(vec, ctx.time, {}, [](const auto& a) { return a.first.time; });
            if (lower == vec.end()) {
                return vec.back().first;
            } else {
                if (lower->first.time == ctx.time) {
                    return lower->first;
                } else if (lower != vec.begin()) {
                    --lower;
                    return lower->first;
                }
            }
        }
        return std::nullopt;
    }

    [[nodiscard]] inline std::optional<property_map> getBestMatchStoredParameters(const SettingsCtx& ctx) const {
        const auto bestMatchSettingsCtx = findBestMatchSettingsCtx(ctx);
        if (bestMatchSettingsCtx == std::nullopt) {
            return std::nullopt;
        }
        const auto& vec        = _storedParameters[bestMatchSettingsCtx.value().context];
        const auto  parameters = std::ranges::find_if(vec, [&](const auto& pair) { return pair.first == bestMatchSettingsCtx.value(); });

        return parameters != vec.end() ? std::optional(parameters->second) : std::nullopt;
    }

    [[nodiscard]] inline std::optional<std::set<std::string>> getBestMatchAutoUpdateParameters(const SettingsCtx& ctx) const {
        const auto bestMatchSettingsCtx = findBestMatchSettingsCtx(ctx);
        if (bestMatchSettingsCtx == std::nullopt || !_autoUpdateParameters.contains(bestMatchSettingsCtx.value())) {
            return std::nullopt;
        } else {
            return _autoUpdateParameters.at(bestMatchSettingsCtx.value());
        }
    }

    NO_INLINE void resolveDuplicateTimestamp(SettingsCtx& ctx) {
        const auto vecIt = _storedParameters.find(ctx.context);
        if (vecIt == _storedParameters.end() || vecIt->second.empty()) {
            return;
        }
        const auto&       vec       = vecIt->second;
        const std::size_t tolerance = 1000; // ns
        // find the last context in sorted vector such that `ctx.time <= ctxToFind <= ctx.time + tolerance`
        const auto lower = std::ranges::lower_bound(vec, ctx.time, {}, [](const auto& elem) { return elem.first.time; });
        const auto upper = std::ranges::upper_bound(vec, ctx.time + tolerance, {}, [](const auto& elem) { return elem.first.time; });
        if (lower != upper && lower != vec.end()) {
            ctx.time = (*(upper - 1)).first.time + 1;
        }
    }

    [[nodiscard]] NO_INLINE property_map setStagedImpl(const property_map& parameters) {
        property_map ret;
        if constexpr (refl::is_reflectable<TBlock>()) {
            for (const auto& [key, value] : parameters) {
                bool isSet            = false;
                auto processOneMember = [&, this]<typename TFieldMeta>(TFieldMeta member) {
                    using Type = unwrap_if_wrapped_t<std::remove_cvref_t<typename TFieldMeta::value_type>>;
                    if constexpr (settings::isWritableMember<Type>(member)) {
                        const auto fieldName = std::string_view(get_display_name(member));
                        if (fieldName == key && std::holds_alternative<Type>(value)) {
                            _stagedParameters.insert_or_assign(key, value);
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
                processMembers(processOneMember);
                if (!isSet) {
                    ret.insert_or_assign(key, pmtv::pmt(value));
                }
            }
        }
        if (!_stagedParameters.empty()) {
            setChanged(true);
        }
        return ret; // N.B. returns those <key:value> parameters that could not be set
    }

    NO_INLINE void addStoredParameters(const property_map& newParameters, const SettingsCtx& ctx) {
        if (!_autoUpdateParameters.contains(ctx)) {
            _autoUpdateParameters[ctx] = getBestMatchAutoUpdateParameters(ctx).value_or(_allWritableMembers);
        }

        std::vector<std::pair<SettingsCtx, property_map>>& sortedVectorForContext = _storedParameters[ctx.context];
        // binary search and merge-sort
        auto it = std::ranges::lower_bound(sortedVectorForContext, ctx.time, std::less<>{}, [](const auto& pair) { return pair.first.time; });
        sortedVectorForContext.insert(it, {ctx, newParameters});
    }

    NO_INLINE void removeExpiredStoredParameters() {
        const auto removeFromAutoUpdateParameters = [this](const auto& begin, const auto& end) {
            for (auto it = begin; it != end; it++) {
                _autoUpdateParameters.erase(it->first);
            }
        };
        std::uint64_t now = settings::convertTimePointToUint64Ns(std::chrono::system_clock::now());
#ifdef __EMSCRIPTEN__
        now += _timePrecisionTolerance;
#endif
        for (auto& [ctx, vec] : _storedParameters) {
            // remove all expired parameters
            if (expiry_time != std::numeric_limits<std::uint64_t>::max()) {
                const auto [first, last] = std::ranges::remove_if(vec, [&](const auto& elem) { return elem.first.time + expiry_time <= now; });
                removeFromAutoUpdateParameters(first, last);
                vec.erase(first, last);
            }

            if (vec.empty()) {
                continue;
            }
            // always keep at least one past parameter set
            auto lower = std::ranges::lower_bound(vec, now, {}, [](const auto& elem) { return elem.first.time; });
            if (lower == vec.end()) {
                removeFromAutoUpdateParameters(vec.begin(), vec.end() - 1);
                vec.erase(vec.begin(), vec.end() - 1);
            } else {
                if (lower->first.time == now) {
                    removeFromAutoUpdateParameters(vec.begin(), lower);
                    vec.erase(vec.begin(), lower);
                } else if (lower != vec.begin() && lower - 1 != vec.begin()) {
                    removeFromAutoUpdateParameters(vec.begin(), lower - 1);
                    vec.erase(vec.begin(), lower - 1);
                }
            }
        }
    }

    [[nodiscard]] NO_INLINE bool isContextPresentInTag(const Tag& tag) const {
        if (tag.map.contains(gr::tag::TRIGGER_META_INFO.shortKey())) {
            const pmtv::pmt& pmtMetaInfo = tag.map.at(std::string(gr::tag::TRIGGER_META_INFO.shortKey()));
            if (std::holds_alternative<property_map>(pmtMetaInfo)) {
                const property_map& metaInfo = std::get<property_map>(pmtMetaInfo);
                if (metaInfo.contains(std::string(gr::tag::CONTEXT.shortKey()))) {
                    return true;
                }
            }
        }
        return false;
    }

    [[nodiscard]] NO_INLINE bool isTriggeredTimePresentInTag(const Tag& tag) const {
        if (tag.map.contains(gr::tag::TRIGGER_TIME.shortKey())) {
            const pmtv::pmt& pmtTimeUtcNs = tag.map.at(std::string(gr::tag::TRIGGER_TIME.shortKey()));
            if (std::holds_alternative<uint64_t>(pmtTimeUtcNs)) {
                return true;
            }
        }
        return false;
    }

    [[nodiscard]] NO_INLINE std::optional<SettingsCtx> createSettingsCtxFromTag(const Tag& tag) const {
        // If TRIGGER_META_INFO or CONTEXT is not present then return std::nullopt
        // IF TRIGGER_TIME is not present then time = now()

        if (isContextPresentInTag(tag)) {
            SettingsCtx      ctx{};
            const pmtv::pmt& pmtMetaInfo = tag.map.at(std::string(gr::tag::TRIGGER_META_INFO.shortKey()));
            ctx.context                  = std::get<property_map>(pmtMetaInfo).at(std::string(gr::tag::CONTEXT.shortKey()));

            // update trigger time if present
            if (isTriggeredTimePresentInTag(tag)) {
                ctx.time = std::get<uint64_t>(tag.map.at(std::string(gr::tag::TRIGGER_TIME.shortKey())));
            }
            if (ctx.time == 0ULL) {
                ctx.time = settings::convertTimePointToUint64Ns(std::chrono::system_clock::now());
            }
            return ctx;
        } else {
            return std::nullopt;
        }
    }

    NO_INLINE void storeCurrentParameters(property_map& parameters) {
        // take a copy of the field -> map value of the old settings
        if constexpr (refl::is_reflectable<TBlock>()) {
            auto processOneMember = [&, this]<typename TFieldMeta>(TFieldMeta member) {
                using Type = unwrap_if_wrapped_t<std::remove_cvref_t<typename TFieldMeta::value_type>>;

                if constexpr (settings::isReadableMember<Type>(member)) {
                    parameters.insert_or_assign(get_display_name(member), pmtv::pmt(member(*_block)));
                }
            };
            processMembers(processOneMember);
        }
    }

}; // class CtxSettings

} // namespace gr

namespace std {
template<>
struct hash<gr::SettingsCtx> {
    [[nodiscard]] size_t operator()(const gr::SettingsCtx& ctx) const noexcept { return ctx.hash(); }
};
} // namespace std

#undef NO_INLINE

#endif // GNURADIO_SETTINGS_HPP
