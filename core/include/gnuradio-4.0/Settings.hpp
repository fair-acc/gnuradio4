#ifndef GNURADIO_SETTINGS_HPP
#define GNURADIO_SETTINGS_HPP

#include <atomic>
#include <chrono>
#include <concepts>
#include <mutex>
#include <optional>
#include <set>
#include <variant>

#include <pmtv/base64/base64.h>
#include <pmtv/pmt.hpp>

#include <format>

#include <gnuradio-4.0/BlockTraits.hpp>
#include <gnuradio-4.0/PmtTypeHelpers.hpp>
#include <gnuradio-4.0/Tag.hpp>
#include <gnuradio-4.0/annotated.hpp>
#include <gnuradio-4.0/meta/formatter.hpp>
#include <gnuradio-4.0/meta/immutable.hpp>
#include <gnuradio-4.0/meta/reflection.hpp>

#if defined(__clang__)
#define NO_INLINE [[gnu::noinline]]
#elif defined(__GNUC__)
#define NO_INLINE [[gnu::noinline, gnu::noipa]]
#else
#define NO_INLINE
#endif

namespace gr {

namespace settings {

template<typename T>
constexpr bool isSupportedVectorType() {
    if constexpr (gr::meta::vector_type<T>) {
        using ValueType = typename T::value_type;
        return std::is_arithmetic_v<ValueType> || std::is_same_v<ValueType, std::string> || std::is_same_v<ValueType, std::complex<double>> || std::is_same_v<ValueType, std::complex<float>> || std::is_enum_v<ValueType>;
    } else {
        return false;
    }
}

template<typename T>
constexpr bool isReadableMember() {
    auto isReadableImmutable = [] {
        if constexpr (gr::meta::is_immutable<T>{}) {
            return isReadableMember<typename T::value_type>();

        } else {
            return false;
        }
    };
    return std::is_arithmetic_v<T> || std::is_same_v<T, std::string> || isSupportedVectorType<T>() || std::is_same_v<T, property_map> //
           || std::is_same_v<T, std::complex<double>> || std::is_same_v<T, std::complex<float>> || std::is_enum_v<T> || isReadableImmutable();
}

template<typename T, typename TMember>
constexpr bool isWritableMember() {
    return isReadableMember<T>() && !std::is_const_v<T> && !std::is_const_v<TMember> && !gr::meta::is_immutable<TMember>{};
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
constexpr std::size_t hash_combine(std::size_t seed, const T& v) noexcept {
    std::hash<T> hasher;
    seed ^= hasher(v) + 0x9e3779b9UZ + (seed << 6) + (seed >> 2);
    return seed;
}

std::size_t computeHash(const pmtv::pmt_var_t& value);

struct PmtHashVisitor {
    template<typename T>
    constexpr std::size_t operator()(const T& value) const {
        if constexpr (std::is_same_v<T, std::monostate>) {
            return 0x9e3779b9UZ; // arbitrary constant seed
        } else if constexpr (pmtv::Scalar<T>) {
            if constexpr (gr::meta::complex_like<T>) {
                using value_t           = typename T::value_type;
                std::size_t        seed = std::hash<value_t>()(value.real());
                std::hash<value_t> hasher;
                seed ^= hasher(value.imag()) + 0x9e3779b9UZ + (seed << 6) + (seed >> 2);
                return seed;
            } else {
                return std::hash<T>()(value);
            }
        } else if constexpr (pmtv::String<T>) {
            return std::hash<std::string>()(value);
        } else if constexpr (gr::meta::vector_type<T>) {
            using value_t    = typename T::value_type;
            std::size_t seed = 0UZ;
            for (const auto& elem : value) {
                if constexpr (pmtv::IsPmt<value_t>) {
                    seed = detail::hash_combine(seed, std::visit(*this, elem));
                } else {
                    seed = detail::hash_combine(seed, (*this)(static_cast<value_t>(elem)));
                }
            }
            return seed;
        } else if constexpr (pmtv::PmtMap<T>) {
            std::size_t seed = 0UZ;
            for (const auto& [key, val] : value) {
                // static_assert(pmtv::IsPmt<decltype(val)>, "val must be a std::variant");
                std::size_t kv_seed = std::hash<std::string>()(key);
                seed                = detail::hash_combine(kv_seed, computeHash(val));
                seed                = detail::hash_combine(seed, kv_seed);
            }
            return seed;
        } else {
            static_assert(gr::meta::always_false<T>, "Unhandled type in PmtHashVisitor.");
            return 0; // unreachable
        }
    }
};

inline std::size_t computeHash(const pmtv::pmt& value) { return std::visit(PmtHashVisitor{}, value); }

template<typename T, typename U = unwrap_if_wrapped_t<std::remove_cvref_t<T>>>
constexpr bool isEnumOrAnnotatedEnum = std::is_enum_v<U>;

template<typename T>
requires isEnumOrAnnotatedEnum<T>
std::expected<T, std::string> tryExtractEnumValue(const pmtv::pmt& pmt, std::string_view key) {
    if (!std::holds_alternative<std::string>(pmt)) {
        return std::unexpected(std::format("Field '{}' expects enum string, got different type", key));
    }

    const std::string& str = std::get<std::string>(pmt);
    if (auto opt = magic_enum::enum_cast<T>(str); opt.has_value()) {
        return *opt;
    }

    return std::unexpected(std::format("Invalid enum value '{}' for key '{}'", str, key));
}

template<typename T, typename U = std::remove_cvref_t<T>>
requires isEnumOrAnnotatedEnum<U>
std::string enumToString(T&& enum_value) {
    if constexpr (is_annotated<U>()) {
        return std::string(magic_enum::enum_name(enum_value.value));
    } else {
        return std::string(magic_enum::enum_name(enum_value));
    }
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

    [[nodiscard]] std::size_t hash() const noexcept { return detail::hash_combine(std::hash<std::uint64_t>()(time), detail::computeHash(context)); }
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

namespace settings {
/**
 * @brief Convert the given `value` to type `T`. If conversion fails or return diagnostic text.
 */
template<typename T>
[[nodiscard]] std::expected<T, std::string> convertParameter(std::string_view key, const pmtv::pmt& value) {
    if constexpr (std::is_enum_v<T>) {
        return detail::tryExtractEnumValue<T>(value, key);
    } else {
        constexpr bool strictChecks = false;
        auto           converted    = pmtv::convert_safely<T, strictChecks>(value);
        if (!converted) {
            return std::unexpected(std::vformat("value for key '{}' has wrong type or can't be converted: {}", std::make_format_args(key, converted.error())));
        }
        return converted;
    }
}

} // namespace settings

struct SettingsBase {
    struct CtxSettingsPair {
        SettingsCtx  context;
        property_map settings;
    };
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
     * @brief return the name of the active context
     */
    [[nodiscard]] virtual const SettingsCtx& activeContext() const noexcept = 0;

    /**
     * @brief removes the given context
     * @return true on success
     */
    [[nodiscard]] virtual bool removeContext(SettingsCtx ctx) = 0;

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
    [[nodiscard]] virtual std::map<pmtv::pmt, std::vector<CtxSettingsPair>, settings::PMTCompare> getStoredAll() const noexcept = 0;

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

    /**
     * @brief Loads parameters from a property_map by matching pmt keys to TBlock's writable data members.
     * Handles type conversion and special cases, such as std::vector<bool>.
     */
    virtual void loadParametersFromPropertyMap(const property_map& parameters, SettingsCtx ctx = {}) = 0;

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
    mutable std::map<pmtv::pmt, std::vector<CtxSettingsPair>, settings::PMTCompare> _storedParameters{};
    property_map                                                                    _defaultParameters{};
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

        if constexpr (refl::reflectable<TBlock>) {
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
            refl::for_each_data_member_index<TBlock>([&](auto kIdx) {
                using MemberType = refl::data_member_type<TBlock, kIdx>;
                using RawType    = std::remove_cvref_t<MemberType>;
                using Type       = unwrap_if_wrapped_t<RawType>;
                auto memberName  = std::string(refl::data_member_name<TBlock, kIdx>.view());

                if constexpr (hasMetaInfo && AnnotatedType<RawType>) {
                    _block->meta_information.value[memberName + "::description"]   = std::string(RawType::description());
                    _block->meta_information.value[memberName + "::documentation"] = std::string(RawType::documentation());
                    _block->meta_information.value[memberName + "::unit"]          = std::string(RawType::unit());
                    _block->meta_information.value[memberName + "::visible"]       = RawType::visible();
                }

                if constexpr (settings::isWritableMember<Type, MemberType>()) {
                    _allWritableMembers.emplace(std::move(memberName));
                }
            });
        }
        _autoForwardParameters.insert(gr::tag::kDefaultTags.begin(), gr::tag::kDefaultTags.end());
    }

    // Not safe as CtxSettings has a pointer back to the block
    // that owns it
    CtxSettings(const CtxSettings& other)            = delete;
    CtxSettings(CtxSettings&& other)                 = delete;
    CtxSettings& operator=(const CtxSettings& other) = delete;
    CtxSettings& operator=(CtxSettings&& other)      = delete;

    CtxSettings(TBlock& block, const CtxSettings& other) {
        _block = std::addressof(block);
        assignFrom(other);
    }

    CtxSettings(TBlock& block, CtxSettings&& other) noexcept {
        _block = std::addressof(block);
        assignFrom(std::move(other));
    }

    void assignFrom(const CtxSettings& other) {
        std::scoped_lock lock(_mutex, other._mutex);
        std::atomic_store_explicit(&_changed, std::atomic_load_explicit(&other._changed, std::memory_order_acquire), std::memory_order_release);
        _storedParameters      = other._storedParameters;
        _defaultParameters     = other._defaultParameters;
        _initBlockParameters   = other._initBlockParameters;
        _allWritableMembers    = other._allWritableMembers;
        _autoUpdateParameters  = other._autoUpdateParameters;
        _autoForwardParameters = other._autoForwardParameters;
        _matchPred             = other._matchPred;
        _activeCtx             = other._activeCtx;
        _stagedParameters      = other._stagedParameters;
        _activeParameters      = other._activeParameters;
    }

    void assignFrom(CtxSettings&& other) noexcept {
        std::scoped_lock lock(_mutex, other._mutex);
        std::atomic_store_explicit(&_changed, std::atomic_load_explicit(&other._changed, std::memory_order_acquire), std::memory_order_release);
        _storedParameters      = std::move(other._storedParameters);
        _defaultParameters     = std::move(other._defaultParameters);
        _initBlockParameters   = std::move(other._initBlockParameters);
        _allWritableMembers    = std::move(other._allWritableMembers);
        _autoUpdateParameters  = std::move(other._autoUpdateParameters);
        _autoForwardParameters = std::move(other._autoForwardParameters);
        _matchPred             = std::exchange(other._matchPred, settings::nullMatchPred);
        _activeCtx             = std::exchange(other._activeCtx, {});
        _stagedParameters      = std::move(other._stagedParameters);
        _activeParameters      = std::move(other._activeParameters);
    }

public:
    [[nodiscard]] bool changed() const noexcept override { return _changed; }

    void setChanged(bool b) noexcept override { _changed.store(b); }

    void setInitBlockParameters(const property_map& parameters) override { _initBlockParameters = parameters; }

    NO_INLINE void init() override {
        storeDefaults();

        if (const property_map failed = set(_initBlockParameters); !failed.empty()) {
            throw gr::exception(std::format("settings could not be applied: {}", failed));
        }

        if (const auto failed = activateContext(); failed == std::nullopt) {
            throw gr::exception("Settings for context could not be activated");
        }
    }

    [[nodiscard]] property_map set(const property_map& parameters, SettingsCtx ctx = {}) override {
        property_map ret;
        if constexpr (refl::reflectable<TBlock>) {
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
                bool isSet = false;
                refl::for_each_data_member_index<TBlock>([&](auto kIdx) {
                    using MemberType = refl::data_member_type<TBlock, kIdx>;
                    using Type       = unwrap_if_wrapped_t<std::remove_cvref_t<MemberType>>;
                    if constexpr (settings::isWritableMember<Type, MemberType>()) {
                        const auto fieldName = refl::data_member_name<TBlock, kIdx>.view();
                        if (fieldName != key) {
                            return;
                        }
                        if (auto convertedValue = settings::convertParameter<Type>(key, value); convertedValue) [[likely]] {
                            if (currentAutoUpdateParameters.contains(key)) {
                                currentAutoUpdateParameters.erase(key);
                            }
                            if constexpr (detail::isEnumOrAnnotatedEnum<Type>) {
                                newParameters.insert_or_assign(key, detail::enumToString(convertedValue.value()));
                            } else {
                                newParameters.insert_or_assign(key, convertedValue.value());
                            }
                            isSet = true;
                        } else {
                            throw gr::exception(convertedValue.error());
                        }
                    }
                });
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

    [[nodiscard]] NO_INLINE const SettingsCtx& activeContext() const noexcept override { return _activeCtx; }

    [[nodiscard]] NO_INLINE bool removeContext(SettingsCtx ctx) override {
        if (ctx.context == "") {
            return false; // Forbid removing default context
        }

        auto it = _storedParameters.find(ctx.context);
        if (it == _storedParameters.end()) {
            return false;
        }

        if (ctx.time == 0ULL) {
            ctx.time = settings::convertTimePointToUint64Ns(std::chrono::system_clock::now());
#ifdef __EMSCRIPTEN__
            ctx.time += _timePrecisionTolerance;
#endif
        }

        std::vector<CtxSettingsPair>& vec     = it->second;
        auto                          exactIt = std::find_if(vec.begin(), vec.end(), [&ctx](const auto& pair) { return pair.context.time == ctx.time; });

        if (exactIt == vec.end()) {
            return false;
        }
        vec.erase(exactIt);

        if (vec.empty()) {
            _storedParameters.erase(ctx.context);
        }

        if (_activeCtx.context == ctx.context) {
            std::ignore = activateContext(); // Activate default context
        }

        return true;
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
        if constexpr (refl::reflectable<TBlock>) {
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
                refl::for_each_data_member_index<TBlock>([&](auto kIdx) {
                    using MemberType = refl::data_member_type<TBlock, kIdx>;
                    using Type       = unwrap_if_wrapped_t<std::remove_cvref_t<MemberType>>;
                    if constexpr (settings::isWritableMember<Type, MemberType>()) {
                        if constexpr (std::is_enum_v<Type>) {
                            if (refl::data_member_name<TBlock, kIdx>.view() == key && autoUpdateParameters->second.contains(key) && std::holds_alternative<std::string>(value)) {
                                _stagedParameters.insert_or_assign(key, value);
                                wasChanged = true;
                            }
                        } else {
                            if (refl::data_member_name<TBlock, kIdx>.view() == key && autoUpdateParameters->second.contains(key) && std::holds_alternative<Type>(value)) {
                                _stagedParameters.insert_or_assign(key, value);
                                wasChanged = true;
                            }
                        }
                    }
                });
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

    [[nodiscard]] std::map<pmtv::pmt, std::vector<CtxSettingsPair>, settings::PMTCompare> getStoredAll() const noexcept override { return _storedParameters; }

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
        if constexpr (refl::reflectable<TBlock>) {
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
                refl::for_each_data_member_index<TBlock>([&](auto kIdx) {
                    using MemberType = refl::data_member_type<TBlock, kIdx>;
                    using RawType    = std::remove_cvref_t<MemberType>;
                    using Type       = unwrap_if_wrapped_t<RawType>;

                    if constexpr (settings::isWritableMember<Type, MemberType>()) {
                        if (refl::data_member_name<TBlock, kIdx>.view() != key) {
                            return;
                        }
                        auto& member = refl::data_member<kIdx>(*_block);

                        std::expected<Type, std::string> maybe_value;
                        if constexpr (detail::isEnumOrAnnotatedEnum<RawType>) {
                            maybe_value = detail::tryExtractEnumValue<Type>(stagedValue, key);
                        } else {
                            maybe_value = std::get<Type>(stagedValue);
                        }

                        if constexpr (is_annotated<RawType>()) {
                            if (maybe_value && member.validate_and_set(*maybe_value)) {
                                result.appliedParameters.insert_or_assign(key, stagedValue);
                                if constexpr (HasSettingsChangedCallback<TBlock>) {
                                    staged.insert_or_assign(key, stagedValue);
                                }
                            } else {
                                std::fputs(std::format("Failed to validate field '{}' with value '{}'.\n", key, stagedValue).c_str(), stderr);
                            }
                        } else {
                            if (!maybe_value) {
                                std::fputs(std::format("Failed to convert key '{}': {}\n", key, maybe_value.error()).c_str(), stderr);
                                return;
                            }
                            member = *maybe_value;
                            result.appliedParameters.insert_or_assign(key, stagedValue);
                            if constexpr (HasSettingsChangedCallback<TBlock>) {
                                staged.insert_or_assign(key, stagedValue);
                            }
                        }

                        if (_autoForwardParameters.contains(key)) {
                            result.forwardParameters.insert_or_assign(key, stagedValue);
                        }
                    }
                });
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

            updateActiveParametersImpl();

            // Update sample_rate if the block performs decimation or interpolation
            if constexpr (TBlock::ResamplingControl::kEnabled) {
                if (result.forwardParameters.contains(gr::tag::SAMPLE_RATE.shortKey()) && (_block->input_chunk_size != 1ULL || _block->output_chunk_size != 1ULL)) {
                    const float ratio         = static_cast<float>(_block->output_chunk_size) / static_cast<float>(_block->input_chunk_size);
                    const float newSampleRate = ratio * std::get<float>(_activeParameters.at(gr::tag::SAMPLE_RATE.shortKey()));
                    result.forwardParameters.insert_or_assign(gr::tag::SAMPLE_RATE.shortKey(), newSampleRate);
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
        if constexpr (refl::reflectable<TBlock>) {
            std::lock_guard lg(_mutex);
            updateActiveParametersImpl();
        }
    }

    NO_INLINE void loadParametersFromPropertyMap(const property_map& parameters, SettingsCtx ctx = {}) override {
        property_map newProperties;

        for (const auto& [key, value] : parameters) {
            bool isSet = false;
            refl::for_each_data_member_index<TBlock>([&](auto kIdx) {
                using MemberType = refl::data_member_type<TBlock, kIdx>;
                using Type       = unwrap_if_wrapped_t<std::remove_cvref_t<MemberType>>;
                if constexpr (settings::isWritableMember<Type, MemberType>()) {
                    const auto fieldName = refl::data_member_name<TBlock, kIdx>.view();
                    if (!isSet && fieldName == key) {
                        newProperties[key] = value;
                        isSet              = true;
                    }
                }
            });

            if (!isSet) {
                if (ctx.context == "") { // store meta_information only for default
                    _block->meta_information[key] = value;
                }
            }
        }

        if (const property_map failed = set(newProperties, ctx); !failed.empty()) {
            throw gr::exception(std::format("settings from property_map could not be loaded: {}", failed));
        }
    }

private:
    NO_INLINE void updateActiveParametersImpl() noexcept {
        refl::for_each_data_member_index<TBlock>([&, this](auto kIdx) {
            using MemberType   = refl::data_member_type<TBlock, kIdx>;
            using RawType      = std::remove_cvref_t<MemberType>;
            using Type         = unwrap_if_wrapped_t<RawType>;
            const auto& member = refl::data_member<kIdx>(*_block);
            const auto& key    = std::string(refl::data_member_name<TBlock, kIdx>.view());

            if constexpr (settings::isReadableMember<Type>()) {
                if constexpr (detail::isEnumOrAnnotatedEnum<RawType>) {
                    _activeParameters.insert_or_assign(key, pmtv::pmt(detail::enumToString(member)));
                } else {
                    _activeParameters.insert_or_assign(key, pmtv::pmt(member));
                }
            }
        });
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
        if (ctx.time == 0ULL || vec.back().context.time <= ctx.time) {
            return vec.back().context;
        } else {
            auto lower = std::ranges::lower_bound(vec, ctx.time, {}, [](const auto& a) { return a.context.time; });
            if (lower == vec.end()) {
                return vec.back().context;
            } else {
                if (lower->context.time == ctx.time) {
                    return lower->context;
                } else if (lower != vec.begin()) {
                    --lower;
                    return lower->context;
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
        const auto  parameters = std::ranges::find_if(vec, [&](const CtxSettingsPair& contextSettings) { return contextSettings.context == bestMatchSettingsCtx.value(); });

        return parameters != vec.end() ? std::optional(parameters->settings) : std::nullopt;
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
        const auto lower = std::ranges::lower_bound(vec, ctx.time, {}, [](const auto& elem) { return elem.context.time; });
        const auto upper = std::ranges::upper_bound(vec, ctx.time + tolerance, {}, [](const auto& elem) { return elem.context.time; });
        if (lower != upper && lower != vec.end()) {
            ctx.time = (*(upper - 1)).context.time + 1;
        }
    }

    [[nodiscard]] NO_INLINE property_map setStagedImpl(const property_map& parameters) {
        property_map ret;
        if constexpr (refl::reflectable<TBlock>) {
            for (const auto& [key, value] : parameters) {
                bool isSet = false;
                refl::for_each_data_member_index<TBlock>([&, this](auto kIdx) {
                    using MemberType = refl::data_member_type<TBlock, kIdx>;
                    using Type       = unwrap_if_wrapped_t<std::remove_cvref_t<MemberType>>;
                    if constexpr (settings::isWritableMember<Type, MemberType>()) {
                        const auto fieldName = refl::data_member_name<TBlock, kIdx>.view();
                        if (fieldName != key) {
                            return;
                        }

                        if (auto convertedValue = settings::convertParameter<Type>(key, value); convertedValue) [[likely]] {
                            if constexpr (detail::isEnumOrAnnotatedEnum<Type>) {
                                _stagedParameters.insert_or_assign(key, detail::enumToString(convertedValue.value()));
                            } else {
                                _stagedParameters.insert_or_assign(key, convertedValue.value());
                            }
                            isSet = true;
                        } else {
                            throw gr::exception(convertedValue.error());
                        }
                    }
                });
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

        std::vector<CtxSettingsPair>& sortedVectorForContext = _storedParameters[ctx.context];
        // binary search and merge-sort
        auto it = std::ranges::lower_bound(sortedVectorForContext, ctx.time, std::less<>{}, [](const auto& pair) { return pair.context.time; });
        sortedVectorForContext.insert(it, {ctx, newParameters});
    }

    NO_INLINE void removeExpiredStoredParameters() {
        const auto removeFromAutoUpdateParameters = [this](const auto& begin, const auto& end) {
            for (auto it = begin; it != end; it++) {
                _autoUpdateParameters.erase(it->context);
            }
        };
        std::uint64_t now = settings::convertTimePointToUint64Ns(std::chrono::system_clock::now());
#ifdef __EMSCRIPTEN__
        now += _timePrecisionTolerance;
#endif
        for (auto& [ctx, vec] : _storedParameters) {
            // remove all expired parameters
            if (expiry_time != std::numeric_limits<std::uint64_t>::max()) {
                const auto [first, last] = std::ranges::remove_if(vec, [&](const auto& elem) { return elem.context.time + expiry_time <= now; });
                removeFromAutoUpdateParameters(first, last);
                vec.erase(first, last);
            }

            if (vec.empty()) {
                continue;
            }
            // always keep at least one past parameter set
            auto lower = std::ranges::lower_bound(vec, now, {}, [](const auto& elem) { return elem.context.time; });
            if (lower == vec.end()) {
                removeFromAutoUpdateParameters(vec.begin(), vec.end() - 1);
                vec.erase(vec.begin(), vec.end() - 1);
            } else {
                if (lower->context.time == now) {
                    removeFromAutoUpdateParameters(vec.begin(), lower);
                    vec.erase(vec.begin(), lower);
                } else if (lower != vec.begin() && lower - 1 != vec.begin()) {
                    removeFromAutoUpdateParameters(vec.begin(), lower - 1);
                    vec.erase(vec.begin(), lower - 1);
                }
            }
        }
    }

    [[nodiscard]] NO_INLINE std::optional<std::string> contextInTag(const Tag& tag) const {
        if (tag.map.contains(gr::tag::CONTEXT.shortKey())) {
            const pmtv::pmt& ctxInfo = tag.map.at(std::string(gr::tag::CONTEXT.shortKey()));
            if (std::holds_alternative<std::string>(ctxInfo)) {
                return std::get<std::string>(ctxInfo);
            }
        }
        return std::nullopt;
    }

    [[nodiscard]] NO_INLINE std::optional<std::uint64_t> triggeredTimeInTag(const Tag& tag) const {
        if (tag.map.contains(gr::tag::TRIGGER_TIME.shortKey())) {
            const pmtv::pmt& pmtTimeUtcNs = tag.map.at(std::string(gr::tag::TRIGGER_TIME.shortKey()));
            if (std::holds_alternative<uint64_t>(pmtTimeUtcNs)) {
                return std::get<uint64_t>(pmtTimeUtcNs);
            }
        }
        return std::nullopt;
    }

    [[nodiscard]] NO_INLINE std::optional<SettingsCtx> createSettingsCtxFromTag(const Tag& tag) const {
        // If CONTEXT is not present then return std::nullopt
        // IF TRIGGER_TIME is not present then time = now()

        if (auto ctxValue = contextInTag(tag); ctxValue.has_value()) {
            SettingsCtx ctx{};
            ctx.context = ctxValue.value();

            // update trigger time if present
            if (auto triggerTime = triggeredTimeInTag(tag); triggerTime.has_value()) {
                ctx.time = triggerTime.value();
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
        if constexpr (refl::reflectable<TBlock>) {
            refl::for_each_data_member_index<TBlock>([&, this](auto kIdx) {
                using MemberType = refl::data_member_type<TBlock, kIdx>;
                using Type       = unwrap_if_wrapped_t<std::remove_cvref_t<MemberType>>;
                if constexpr (settings::isReadableMember<Type>()) {
                    if constexpr (detail::isEnumOrAnnotatedEnum<Type>) {
                        parameters.insert_or_assign(std::string(refl::data_member_name<TBlock, kIdx>.view()), pmtv::pmt(detail::enumToString(refl::data_member<kIdx>(*_block))));
                    } else {
                        parameters.insert_or_assign(std::string(refl::data_member_name<TBlock, kIdx>.view()), pmtv::pmt(refl::data_member<kIdx>(*_block)));
                    }
                }
            });
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
