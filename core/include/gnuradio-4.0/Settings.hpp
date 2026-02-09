#ifndef GNURADIO_SETTINGS_HPP
#define GNURADIO_SETTINGS_HPP

#include <atomic>
#include <chrono>
#include <concepts>
#include <format>
#include <mutex>
#include <optional>
#include <set>
#include <variant>

#include <gnuradio-4.0/BlockTraits.hpp>
#include <gnuradio-4.0/PmtTypeHelpers.hpp>
#include <gnuradio-4.0/Tag.hpp>
#include <gnuradio-4.0/ValueHelper.hpp>
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
constexpr bool isSupportedVectorOrTensorType() {
    if constexpr (gr::meta::vector_type<T> || is_tensor<T>) {
        using ValueType = typename T::value_type;
        return std::is_arithmetic_v<ValueType> || std::is_same_v<ValueType, std::string> || std::is_same_v<ValueType, std::pmr::string> || std::is_same_v<ValueType, std::complex<double>> || std::is_same_v<ValueType, std::complex<float>> || std::is_enum_v<ValueType> || std::is_same_v<ValueType, pmt::Value>;
    } else {
        return false;
    }
}

template<typename T>
constexpr bool isReadableMember() {
    auto isReadableImmutable = [] {
        if constexpr (gr::meta::is_immutable<T>{}) {
            return isReadableMember<typename T::value_type>();
        } else if constexpr (is_annotated<T>{}) {
            return isReadableMember<typename T::value_type>();

        } else {
            return false;
        }
    };
    return std::is_arithmetic_v<T> || std::is_same_v<T, std::string> || isSupportedVectorOrTensorType<T>() || std::is_same_v<T, property_map> //
           || std::is_same_v<T, std::complex<double>> || std::is_same_v<T, std::complex<float>> || std::is_enum_v<T> || std::is_same_v<T, pmt::Value> || isReadableImmutable();
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

inline std::strong_ordering comparePmt(const pmt::Value& lhs, const pmt::Value& rhs) {
    // If the types are different, cast rhs to the type of lhs and compare
    if (lhs.container_type() != rhs.container_type()) {
        // TODO: throw if types are not the same?
        return lhs.container_type() <=> rhs.container_type();
    } else if (lhs.value_type() != rhs.value_type()) {
        return lhs.value_type() <=> rhs.value_type();
    } else {
        if (lhs.holds<std::string_view>()) {
            return lhs.value_or(std::string_view{}) <=> rhs.value_or(std::string_view{});
        } else if (lhs.holds<int>()) {
            return lhs.value_or(0) <=> rhs.value_or(0);
        } else {
            throw gr::exception("Invalid CtxSettings context type " + std::string(typeid(lhs).name()));
        }
    }
}

// pmt::Value comparison is needed to use it as a key of std::map
struct PMTCompare {
    bool operator()(const pmt::Value& lhs, const pmt::Value& rhs) const { return comparePmt(lhs, rhs) == std::strong_ordering::less; }
};

} // namespace settings

struct ApplyStagedParametersResult {
    property_map forwardParameters; // parameters that should be forwarded to dependent child blocks
    property_map appliedParameters;
};

namespace detail {

#ifdef __EMSCRIPTEN__
template<typename TValue>
auto castToGrSizeIfNeeded(const TValue& value) {
    if constexpr (std::is_same_v<TValue, std::size_t>) {
        return static_cast<gr::Size_t>(value);
    } else {
        return value;
    }
};
#else
template<typename TValue>
auto castToGrSizeIfNeeded(const TValue& value) {
    return value;
};
#endif

template<typename T>
auto unwrap_decorated_value(const T& value) {
    if constexpr (AnnotatedType<T>) {
        return castToGrSizeIfNeeded(value.value);
    } else if constexpr (meta::ImmutableType<T>) {
        return castToGrSizeIfNeeded(value.value());
    } else {
        return castToGrSizeIfNeeded(value);
    }
}

template<typename T>
const auto& unwrap_decorated_reference(const T& value) {
    if constexpr (AnnotatedType<T>) {
        return value.value;
    } else if constexpr (meta::ImmutableType<T>) {
        return value.value();
    } else {
        return value;
    }
};

std::size_t computeHash(const pmt::Value& value);

template<class T>
constexpr std::size_t hash_combine(std::size_t seed, const T& v) noexcept {
    std::hash<T> hasher;
    seed ^= hasher(v) + 0x9e3779b9UZ + (seed << 6) + (seed >> 2);
    return seed;
}

inline auto computeValueHash = meta::overloaded([](const std::string_view& sv) { return std::hash<std::string_view>()(sv); }, //
    []<typename T>(const gr::Tensor<T>& tensor) {
        std::size_t seed = 9UZ;
        for (const auto& v : tensor) {
            seed = detail::hash_combine(seed, computeHash(pmt::Value(v)));
        }
        return seed;
    }, //
    [](const gr::property_map& map) {
        std::size_t seed = 0UZ;
        for (const auto& [k, v] : map) {
            std::size_t kv_seed = std::hash<std::string_view>()(k);
            seed                = detail::hash_combine(kv_seed, computeHash(v));
            seed                = detail::hash_combine(seed, kv_seed);
        }
        return seed;
    }, //
    [](const std::monostate) {
        // arbitrary constant seed
        return 0x9e3779b9UZ;
    }, //
    []<typename VT>(const std::complex<VT>& v) {
        std::hash<VT> hasher;
        std::size_t   seed = hasher(v.real());
        seed ^= hasher(v.imag()) + 0x9e3779b9UZ + (seed << 6) + (seed >> 2);
        return seed;
    }, //
    []<typename T>(const T& v) {
        if constexpr (gr::meta::complex_like<std::remove_cvref_t<T>>) {
            using value_t           = typename T::value_type;
            std::size_t        seed = std::hash<value_t>()(v.real());
            std::hash<value_t> hasher;
            seed ^= hasher(v.imag()) + 0x9e3779b9UZ + (seed << 6) + (seed >> 2);
            return seed;
        } else {
            return std::hash<T>()(v);
        }
    });

inline std::size_t computeHash(const pmt::Value& value) {
    std::size_t result = 0UZ;
    pmt::ValueVisitor([&](const auto& v) { result = computeValueHash(v); }).visit(value);
    return result;
}

template<typename TCollection>
auto collectionToTensor(const TCollection& collection) {
    using TValue       = typename TCollection::value_type;
    using TTensorValue = std::conditional_t<std::is_same_v<std::string, TValue>, pmt::Value, TValue>;
    Tensor<TTensorValue> result(extents_from, {collection.size()});
    std::ranges::copy(collection, result.begin());
    return result;
}

template<typename T, typename U = unwrap_if_wrapped_t<std::remove_cvref_t<T>>>
constexpr bool isEnumOrAnnotatedEnum = std::is_enum_v<U>;

template<typename T>
requires isEnumOrAnnotatedEnum<T>
std::expected<T, std::string> tryExtractEnumValue(const pmt::Value& pmt, std::string_view key) {

    auto str = pmt.value_or(std::string_view{});
    if (str.data() == nullptr) {
        return std::unexpected(std::format("Field '{}' expects enum string, got different type", key));
    }

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
    std::uint64_t time    = 0ULL;          // UTC-based time-stamp in ns, time from which the setting is valid, 0U is undefined time
    pmt::Value    context = std::string(); // user-defined multiplexing context for which the setting is valid

    bool operator==(const SettingsCtx&) const = default;

    std::partial_ordering operator<=>(const SettingsCtx& other) const {
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
[[nodiscard]] std::expected<T, std::string> convertParameter(std::string_view key, const pmt::Value& value) {
    if constexpr (std::is_enum_v<T>) {
        return detail::tryExtractEnumValue<T>(value, key);
    } else {
        if constexpr (std::is_same_v<T, std::string>) {
            auto sv = value.value_or(std::string_view{});
            if (sv.data()) {
                return std::string(sv);
            } else {
                return std::unexpected(std::format("value {} for key '{}' has wrong type {} {}, needs {}", value, key, value.value_type(), value.container_type(), std::string(meta::type_name<T>())));
            }

        } else if constexpr (std::is_same_v<T, std::vector<std::string>>) {
            const auto* tensorValue = value.get_if<Tensor<pmt::Value>>();
            if (!tensorValue) {
                return std::unexpected(std::format("Value {} is not a tensor of Value", value));

            } else {
                std::vector<std::string> converted(tensorValue->size());
                std::ranges::transform(*tensorValue, converted.begin(), [](const pmt::Value& in) { return in.value_or(std::string()); });
                return converted;
            }

        } else if constexpr (meta::is_instantiation_of<T, std::vector>) {
            using TValue            = typename T::value_type;
            const auto* tensorValue = value.get_if<Tensor<TValue>>();
            if (!tensorValue) {
                return std::unexpected(std::format("Value {} is not a tensor of {}", value, meta::type_name<TValue>()));

            } else {
                std::vector<TValue> converted(tensorValue->size());
                std::ranges::copy(*tensorValue, converted.begin());
                return converted;
            }

        } else {
            constexpr bool strictChecks = false;
            auto           converted    = pmt::convert_safely<T, strictChecks>(value);
            if (!converted) {
                return std::unexpected(std::format("value {} for key '{}' has wrong type {} {}, needs {}", value, key, value.value_type(), value.container_type(), std::string(meta::type_name<T>())));
            }
            return *converted;
        }
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
    [[nodiscard]] virtual std::optional<pmt::Value> get(const std::string& parameter_key) const noexcept = 0;

    /**
     * @brief return all (or for selected multiple keys) stored block settings for provided context as key-value pairs
     */
    [[nodiscard]] virtual std::optional<property_map> getStored(std::span<const std::string> parameterKeys = {}, SettingsCtx ctx = {}) const noexcept = 0;

    /**
     * @brief return available stored block setting for provided context as key-value pair for a single key
     */
    [[nodiscard]] virtual std::optional<pmt::Value> getStored(const std::string& parameter_key, SettingsCtx ctx = {}) const noexcept = 0;

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
    [[nodiscard]] virtual std::map<pmt::Value, std::vector<CtxSettingsPair>, settings::PMTCompare> getStoredAll() const noexcept = 0;

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

/**
 * @brief Non-templated base class for CtxSettings containing all type-independent data and logic.
 * This is compiled once in Settings.cpp rather than instantiated per block type. (Optimization I)
 */
class CtxSettingsBase : public SettingsBase {
public:
    using MatchPredicate = std::function<std::optional<bool>(const pmt::Value&, const pmt::Value&, std::size_t)>;

protected:
    std::atomic_bool   _changed{false};
    mutable std::mutex _mutex{};

    // key: SettingsCtx.context, value: queue of parameters with the same SettingsCtx.context but for different time
    mutable std::map<pmt::Value, std::vector<CtxSettingsPair>, settings::PMTCompare> _storedParameters{};
    property_map                                                                     _defaultParameters{};
    property_map                                                                     _initBlockParameters{};
    std::map<SettingsCtx, std::set<std::string>>                                     _autoUpdateParameters{};
    std::set<std::string>                                                            _autoForwardParameters{};
    MatchPredicate                                                                   _matchPred = settings::nullMatchPred;
    SettingsCtx                                                                      _activeCtx{};
    property_map                                                                     _stagedParameters{};
    property_map                                                                     _activeParameters{};

    const std::size_t _timePrecisionTolerance = 100; // ns, now used for emscripten

    // Virtual hooks for type-dependent logic called from type-independent methods
    [[nodiscard]] virtual property_map                 doSetStagedImpl(const property_map& parameters) = 0;
    [[nodiscard]] virtual const std::set<std::string>& doGetAllWritableMembers() const                 = 0;

public:
    // Settings configuration
    std::uint64_t expiry_time{std::numeric_limits<std::uint64_t>::max()};

    // --- Type-independent virtual method implementations (defined in Settings.cpp) ---

    [[nodiscard]] bool changed() const noexcept override;
    void               setChanged(bool b) noexcept override;
    void               setInitBlockParameters(const property_map& parameters) override;

    [[nodiscard]] const SettingsCtx& activeContext() const noexcept override;

    [[nodiscard]] std::set<std::string>& autoForwardParameters() noexcept override;
    [[nodiscard]] const property_map&    defaultParameters() const noexcept override;
    [[nodiscard]] const property_map&    activeParameters() const noexcept override;

    [[nodiscard]] property_map              get(std::span<const std::string> parameterKeys = {}) const noexcept override;
    [[nodiscard]] std::optional<pmt::Value> get(const std::string& parameterKey) const noexcept override;

    [[nodiscard]] std::optional<property_map> getStored(std::span<const std::string> parameterKeys = {}, SettingsCtx ctx = {}) const noexcept override;
    [[nodiscard]] std::optional<pmt::Value>   getStored(const std::string& parameterKey, SettingsCtx ctx = {}) const noexcept override;

    [[nodiscard]] gr::Size_t getNStoredParameters() const noexcept override;
    [[nodiscard]] gr::Size_t getNAutoUpdateParameters() const noexcept override;

    [[nodiscard]] std::map<pmt::Value, std::vector<CtxSettingsPair>, settings::PMTCompare> getStoredAll() const noexcept override;

    [[nodiscard]] const property_map& stagedParameters() const override;

    [[nodiscard]] std::set<std::string> autoUpdateParameters(SettingsCtx ctx = {}) noexcept override;

    [[nodiscard]] property_map setStaged(const property_map& parameters) override;

    [[nodiscard]] std::optional<SettingsCtx> activateContext(SettingsCtx ctx = {}) override;
    [[nodiscard]] bool                       removeContext(SettingsCtx ctx) override;

    void assignFrom(const CtxSettingsBase& other);
    void assignFrom(CtxSettingsBase&& other) noexcept;

protected:
    // --- Private helpers (defined in Settings.cpp) ---
    [[nodiscard]] std::optional<pmt::Value>            findBestMatchCtx(const pmt::Value& contextToSearch) const;
    [[nodiscard]] std::optional<SettingsCtx>           findBestMatchSettingsCtx(const SettingsCtx& ctx) const;
    [[nodiscard]] std::optional<property_map>          getBestMatchStoredParameters(const SettingsCtx& ctx) const;
    [[nodiscard]] std::optional<std::set<std::string>> getBestMatchAutoUpdateParameters(const SettingsCtx& ctx) const;
    void                                               resolveDuplicateTimestamp(SettingsCtx& ctx);
    void                                               addStoredParameters(const property_map& newParameters, const SettingsCtx& ctx);
    void                                               removeExpiredStoredParameters();
    [[nodiscard]] std::optional<std::string>           contextInTag(const Tag& tag) const;
    [[nodiscard]] std::optional<std::uint64_t>         triggeredTimeInTag(const Tag& tag) const;
    [[nodiscard]] std::optional<SettingsCtx>           createSettingsCtxFromTag(const Tag& tag) const;
}; // class CtxSettingsBase

template<typename TBlock>
class CtxSettings : public CtxSettingsBase {
    TBlock* _block = nullptr;

    // Virtual hook: delegates to type-dependent setStagedImpl using static dispatch table
    [[nodiscard]] property_map doSetStagedImpl(const property_map& parameters) override {
        property_map ret;
        if constexpr (refl::reflectable<TBlock>) {
            const auto& setters = parameterSetters();
            for (const auto& [key, value] : parameters) {
                auto it = setters.find(key);
                if (it != setters.end()) {
                    if (auto error = it->second(key, value, _stagedParameters)) {
                        throw gr::exception(*error);
                    }
                } else {
                    ret.insert_or_assign(key, value);
                }
            }
        }
        if (!_stagedParameters.empty()) {
            setChanged(true);
        }
        return ret;
    }

    // Virtual hook: returns the static allWritableMembers set for this block type
    [[nodiscard]] const std::set<std::string>& doGetAllWritableMembers() const override { return allWritableMembers(); }

public:
    // Static function - computed once per block type (Optimization B)
    [[nodiscard]] static const std::set<std::string>& allWritableMembers() {
        static const std::set<std::string> members = [] {
            std::set<std::string> result;
            if constexpr (refl::reflectable<TBlock>) {
                refl::for_each_data_member_index<TBlock>([&](auto kIdx) {
                    using MemberType = refl::data_member_type<TBlock, kIdx>;
                    using RawType    = std::remove_cvref_t<MemberType>;
                    using Type       = unwrap_if_wrapped_t<RawType>;
                    if constexpr (settings::isWritableMember<Type, MemberType>()) {
                        result.emplace(std::string(refl::data_member_name<TBlock, kIdx>.view()));
                    }
                });
            }
            return result;
        }();
        return members;
    }

    // ===== Static dispatch tables for compile-time optimization (Optimization F) =====

    // Type aliases for dispatch function pointers
    using ParameterSetter       = std::optional<std::string> (*)(std::string_view key, const pmt::Value& value, property_map& newParameters);
    using StagedParameterSetter = std::optional<std::string> (*)(std::string_view key, const pmt::Value& value, property_map& stagedParameters);
    using AutoUpdateHandler     = bool (*)(std::string_view key, const pmt::Value& value, const std::set<std::string>& autoUpdateParams, property_map& stagedParameters);
    using StagedApplier         = bool (*)(TBlock* block, std::string_view key, const pmt::Value& value, property_map& applied, property_map& staged, bool hasCallback);
    using ParameterReader       = void (*)(const TBlock* block, property_map& parameters);
    using ActiveParameterReader = void (*)(const TBlock* block, property_map& activeParameters);

private:
    // Helper template for parameter setting (set method) - instantiated once per member type
    template<typename Type>
    static std::optional<std::string> setParameterImpl(std::string_view key, const pmt::Value& value, property_map& newParameters) {
        if (auto convertedValue = settings::convertParameter<Type>(key, value); convertedValue) [[likely]] {
            const auto keyStr = std::pmr::string(key);
            if constexpr (detail::isEnumOrAnnotatedEnum<Type>) {
                newParameters.insert_or_assign(keyStr, detail::enumToString(convertedValue.value()));
            } else if constexpr (meta::is_instantiation_of<Type, std::vector>) {
                newParameters.insert_or_assign(keyStr, pmt::Value(detail::collectionToTensor(*convertedValue)));
            } else {
                newParameters.insert_or_assign(keyStr, detail::castToGrSizeIfNeeded(convertedValue.value()));
            }
            return std::nullopt; // success
        } else {
            return convertedValue.error(); // error message
        }
    }

    // Helper template for autoUpdate - checks type compatibility and updates staged parameters
    template<typename Type>
    static bool autoUpdateImpl(std::string_view key, const pmt::Value& value, const std::set<std::string>& autoUpdateParams, property_map& stagedParameters) {
        const auto keyStr = std::string(key);
        if (!autoUpdateParams.contains(keyStr)) {
            return false;
        }
        const auto keyPmr = std::pmr::string(key);
        if constexpr (std::is_enum_v<Type>) {
            if (value.holds<std::string>()) {
                stagedParameters.insert_or_assign(keyPmr, value);
                return true;
            }
#ifdef __EMSCRIPTEN__
        } else if constexpr (std::is_same_v<Type, std::size_t> && !std::is_same_v<std::size_t, gr::Size_t>) {
            if (value.holds<gr::Size_t>()) {
                stagedParameters.insert_or_assign(keyPmr, value);
                return true;
            }
#endif
        } else if constexpr (std::is_same_v<Type, std::vector<std::string>>) {
            using TValue = std::pmr::string;
            if (value.holds<Tensor<TValue>>()) {
                auto vectorValue = pmt::convertTo<Tensor<TValue>>(value);
                stagedParameters.insert_or_assign(keyPmr, std::move(vectorValue.value()));
                return true;
            }
        } else if constexpr (meta::is_instantiation_of<Type, std::vector>) {
            using TValue = typename Type::value_type;
            if (value.holds<Tensor<TValue>>()) {
                auto vectorValue = pmt::convertTo<Tensor<TValue>>(value);
                stagedParameters.insert_or_assign(keyPmr, std::move(vectorValue.value()));
                return true;
            }
        } else {
            if (value.holds<Type>()) {
                stagedParameters.insert_or_assign(keyPmr, value);
                return true;
            }
        }
        return false;
    }

    // Helper template for applyStagedParameters - applies value to block member
    template<std::size_t kIdx, typename RawType, typename Type>
    static bool applyStagedImpl(TBlock* block, std::string_view key, const pmt::Value& stagedValue, property_map& applied, property_map& staged, bool hasCallback) {
        auto&      member = refl::data_member<kIdx>(*block);
        const auto keyPmr = std::pmr::string(key);

        std::expected<Type, std::string> maybe_value;
        if constexpr (detail::isEnumOrAnnotatedEnum<RawType>) {
            maybe_value = detail::tryExtractEnumValue<Type>(stagedValue, key);
        } else if constexpr (std::is_same_v<Type, std::string>) {
            auto str = stagedValue.value_or(std::string_view{});
            if (str.data() != nullptr) {
                maybe_value = std::string(str);
            } else {
                maybe_value = std::unexpected("Unexpected type in stagedValue");
            }
        } else if constexpr (meta::is_instantiation_of<Type, std::vector>) {
            using TValue       = typename Type::value_type;
            using TTensorValue = std::conditional_t<std::is_same_v<std::string, TValue>, pmt::Value, TValue>;
            auto tensor        = checked_access_ptr{stagedValue.get_if<Tensor<TTensorValue>>()};
            if (tensor != nullptr) {
                maybe_value = typename decltype(maybe_value)::value_type{};
                if (auto conversionResult = pmt::assignTo(*maybe_value, *tensor); !conversionResult) {
                    maybe_value = std::unexpected(conversionResult.error().message);
                }
            } else {
                maybe_value = std::unexpected("Unexpected type in stagedValue");
            }
#ifdef __EMSCRIPTEN__
        } else if constexpr (std::is_same_v<Type, std::size_t> && !std::is_same_v<std::size_t, gr::Size_t>) {
            auto ptr = checked_access_ptr{stagedValue.get_if<gr::Size_t>()};
            if (ptr != nullptr) {
                maybe_value = static_cast<std::size_t>(*ptr);
            } else {
                maybe_value = std::unexpected("Unexpected type in stagedValue");
            }
#endif
        } else {
            auto ptr = checked_access_ptr{stagedValue.get_if<Type>()};
            if (ptr != nullptr) {
                maybe_value = *ptr;
            } else {
                maybe_value = std::unexpected("Unexpected type in stagedValue");
            }
        }

        if constexpr (is_annotated<RawType>()) {
            if (maybe_value && member.validate_and_set(*maybe_value)) {
                applied.insert_or_assign(keyPmr, stagedValue);
                if (hasCallback) {
                    staged.insert_or_assign(keyPmr, stagedValue);
                }
                return true;
            } else {
                std::fputs(std::format("Failed to validate field '{}' with value '{}'.\n", std::string_view(key), stagedValue).c_str(), stderr);
                return false;
            }
        } else {
            if (!maybe_value) {
                std::fputs(std::format("Failed to convert key '{}': {}\n", std::string_view(key), maybe_value.error()).c_str(), stderr);
                return false;
            }
            member = *maybe_value;
            applied.insert_or_assign(keyPmr, stagedValue);
            if (hasCallback) {
                staged.insert_or_assign(keyPmr, stagedValue);
            }
            return true;
        }
    }

    // Helper template for storeCurrentParameters - reads member value into property_map
    template<std::size_t kIdx, typename Type>
    static void storeParameterImpl(const TBlock* block, property_map& parameters) {
        const auto& key    = std::pmr::string(refl::data_member_name<TBlock, kIdx>.view());
        const auto& member = refl::data_member<kIdx>(*block);
        if constexpr (detail::isEnumOrAnnotatedEnum<Type>) {
            parameters.insert_or_assign(key, detail::enumToString(member));
        } else if constexpr (meta::is_instantiation_of<Type, std::vector>) {
            const auto& from = detail::unwrap_decorated_value(member);
            parameters.insert_or_assign(key, detail::collectionToTensor(from));
        } else {
            parameters.insert_or_assign(key, detail::unwrap_decorated_value(member));
        }
    }

    // Helper template for updateActiveParameters - reads member value for active parameters
    template<std::size_t kIdx, typename RawType, typename Type>
    static void updateActiveParameterImpl(const TBlock* block, property_map& activeParameters) {
        const auto& key    = std::string(refl::data_member_name<TBlock, kIdx>.view());
        const auto& member = refl::data_member<kIdx>(*block);
        if constexpr (detail::isEnumOrAnnotatedEnum<RawType>) {
            activeParameters.insert_or_assign(convert_string_domain(key), detail::enumToString(member));
        } else if constexpr (meta::is_instantiation_of<Type, std::vector>) {
            const auto& from = detail::unwrap_decorated_reference(member);
            activeParameters.insert_or_assign(convert_string_domain(key), pmt::Value(detail::collectionToTensor(from)));
        } else {
            activeParameters.insert_or_assign(convert_string_domain(key), detail::unwrap_decorated_value(member));
        }
    }

public:
    // Static dispatch table for set() method
    [[nodiscard]] static const std::unordered_map<std::string_view, ParameterSetter>& parameterSetters() {
        static const std::unordered_map<std::string_view, ParameterSetter> setters = [] {
            std::unordered_map<std::string_view, ParameterSetter> result;
            if constexpr (refl::reflectable<TBlock>) {
                refl::for_each_data_member_index<TBlock>([&](auto kIdx) {
                    using MemberType = refl::data_member_type<TBlock, kIdx>;
                    using RawType    = std::remove_cvref_t<MemberType>;
                    using Type       = unwrap_if_wrapped_t<RawType>;
                    if constexpr (settings::isWritableMember<Type, MemberType>()) {
                        constexpr auto fieldName = refl::data_member_name<TBlock, kIdx>;
                        result[fieldName.view()] = &setParameterImpl<Type>;
                    }
                });
            }
            return result;
        }();
        return setters;
    }

    // Static dispatch table for autoUpdate() method
    [[nodiscard]] static const std::unordered_map<std::string_view, AutoUpdateHandler>& autoUpdateHandlers() {
        static const std::unordered_map<std::string_view, AutoUpdateHandler> handlers = [] {
            std::unordered_map<std::string_view, AutoUpdateHandler> result;
            if constexpr (refl::reflectable<TBlock>) {
                refl::for_each_data_member_index<TBlock>([&](auto kIdx) {
                    using MemberType = refl::data_member_type<TBlock, kIdx>;
                    using Type       = unwrap_if_wrapped_t<std::remove_cvref_t<MemberType>>;
                    if constexpr (settings::isWritableMember<Type, MemberType>()) {
                        constexpr auto fieldName = refl::data_member_name<TBlock, kIdx>;
                        result[fieldName.view()] = &autoUpdateImpl<Type>;
                    }
                });
            }
            return result;
        }();
        return handlers;
    }

    // Static dispatch table for applyStagedParameters() method
    [[nodiscard]] static const std::unordered_map<std::string_view, StagedApplier>& stagedAppliers() {
        static const std::unordered_map<std::string_view, StagedApplier> appliers = [] {
            std::unordered_map<std::string_view, StagedApplier> result;
            if constexpr (refl::reflectable<TBlock>) {
                refl::for_each_data_member_index<TBlock>([&](auto kIdx) {
                    using MemberType = refl::data_member_type<TBlock, kIdx>;
                    using RawType    = std::remove_cvref_t<MemberType>;
                    using Type       = unwrap_if_wrapped_t<RawType>;
                    if constexpr (settings::isWritableMember<Type, MemberType>()) {
                        constexpr auto fieldName = refl::data_member_name<TBlock, kIdx>;
                        result[fieldName.view()] = &applyStagedImpl<kIdx, RawType, Type>;
                    }
                });
            }
            return result;
        }();
        return appliers;
    }

    // Static list of parameter readers for storeCurrentParameters()
    [[nodiscard]] static const std::vector<ParameterReader>& parameterReaders() {
        static const std::vector<ParameterReader> readers = [] {
            std::vector<ParameterReader> result;
            if constexpr (refl::reflectable<TBlock>) {
                refl::for_each_data_member_index<TBlock>([&](auto kIdx) {
                    using MemberType = refl::data_member_type<TBlock, kIdx>;
                    using Type       = unwrap_if_wrapped_t<std::remove_cvref_t<MemberType>>;
                    if constexpr (settings::isReadableMember<Type>()) {
                        result.push_back(&storeParameterImpl<kIdx, Type>);
                    }
                });
            }
            return result;
        }();
        return readers;
    }

    // Static list of active parameter readers for updateActiveParameters()
    [[nodiscard]] static const std::vector<ActiveParameterReader>& activeParameterReaders() {
        static const std::vector<ActiveParameterReader> readers = [] {
            std::vector<ActiveParameterReader> result;
            if constexpr (refl::reflectable<TBlock>) {
                refl::for_each_data_member_index<TBlock>([&](auto kIdx) {
                    using MemberType = refl::data_member_type<TBlock, kIdx>;
                    using RawType    = std::remove_cvref_t<MemberType>;
                    using Type       = unwrap_if_wrapped_t<RawType>;
                    if constexpr (settings::isReadableMember<Type>()) {
                        result.push_back(&updateActiveParameterImpl<kIdx, RawType, Type>);
                    }
                });
            }
            return result;
        }();
        return readers;
    }

public:
    explicit CtxSettings(TBlock& block, MatchPredicate matchPred = settings::nullMatchPred) noexcept : CtxSettingsBase(), _block(&block) {
        _matchPred = std::move(matchPred);
        if constexpr (requires { &TBlock::settingsChanged; }) { // if settingsChanged is defined
            static_assert(HasSettingsChangedCallback<TBlock>, "if provided, settingsChanged must have either a `(const property_map& old, property_map& new, property_map& fwd)`"
                                                              "or `(const property_map& old, property_map& new)` paremeter signatures.");
        }

        if constexpr (requires { &TBlock::reset; }) { // if reset is defined
            static_assert(HasSettingsResetCallback<TBlock>, "if provided, reset() may have no function parameters");
        }
        // meta_information population deferred to init() (Optimization B)
        _autoForwardParameters.insert(gr::tag::kDefaultTags.begin(), gr::tag::kDefaultTags.end());
    }

    // Not safe as CtxSettings has a pointer back to the block
    // that owns it
    CtxSettings(const CtxSettings& other)            = delete;
    CtxSettings(CtxSettings&& other)                 = delete;
    CtxSettings& operator=(const CtxSettings& other) = delete;
    CtxSettings& operator=(CtxSettings&& other)      = delete;

    CtxSettings(TBlock& block, const CtxSettings& other) : CtxSettingsBase() {
        _block = std::addressof(block);
        assignFrom(other);
    }

    CtxSettings(TBlock& block, CtxSettings&& other) noexcept : CtxSettingsBase() {
        _block = std::addressof(block);
        assignFrom(std::move(other));
    }

    NO_INLINE void init() override {
        // Populate meta_information at runtime (deferred from constructor - Optimization B)
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
                if constexpr (hasMetaInfo && AnnotatedType<RawType>) {
                    auto  memberName                                                        = std::string(refl::data_member_name<TBlock, kIdx>.view());
                    auto& meta_information                                                  = _block->meta_information;
                    meta_information[convert_string_domain(memberName) + "::description"]   = std::string(RawType::description());
                    meta_information[convert_string_domain(memberName) + "::documentation"] = std::string(RawType::documentation());
                    meta_information[convert_string_domain(memberName) + "::unit"]          = std::string(RawType::unit());
                    meta_information[convert_string_domain(memberName) + "::visible"]       = RawType::visible();
                }
            });
        }

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
                _autoUpdateParameters[ctx] = getBestMatchAutoUpdateParameters(ctx).value_or(allWritableMembers());
            }
            auto& currentAutoUpdateParameters = _autoUpdateParameters[ctx];

            // Use static dispatch table for O(1) lookup instead of O(members) iteration (Optimization F)
            const auto& setters = parameterSetters();
            for (const auto& [key, value] : parameters) {
                if (value.is_monostate()) {
                    continue;
                }

                auto it = setters.find(key);
                if (it != setters.end()) {
                    if (auto error = it->second(key, value, newParameters)) {
                        throw gr::exception(*error);
                    }
                    // Remove from auto-update set if present
                    if (auto autoIt = currentAutoUpdateParameters.find(std::string(key)); autoIt != currentAutoUpdateParameters.end()) {
                        currentAutoUpdateParameters.erase(autoIt);
                    }
                } else {
                    ret.insert_or_assign(key, value);
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

    void storeDefaults() override { this->storeCurrentParameters(_defaultParameters); }

    NO_INLINE void resetDefaults() override {
        // add default parameters to stored and apply the parameters
        auto ctx = SettingsCtx{settings::convertTimePointToUint64Ns(std::chrono::system_clock::now()), std::string()};
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

            const auto autoUpdateParametersIt = _autoUpdateParameters.find(ctx);
            if (autoUpdateParametersIt == _autoUpdateParameters.end()) {
                return;
            }

            // Use static dispatch table for O(1) lookup instead of O(members) iteration (Optimization F)
            const auto& handlers   = autoUpdateHandlers();
            const auto& parameters = tag.map;
            bool        wasChanged = false;
            for (const auto& [key, value] : parameters) {
                auto it = handlers.find(key);
                if (it != handlers.end()) {
                    if (it->second(key, value, autoUpdateParametersIt->second, _stagedParameters)) {
                        wasChanged = true;
                    }
                }
            }

            if (tagCtx == std::nullopt && !wasChanged) { // not context and no parameters in the Tag
                _stagedParameters.clear();
                setChanged(false);
            } else if (activeCtxChanged || wasChanged) {
                setChanged(true);
            }
        }
    }

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
            if (_stagedParameters.contains(static_cast<std::pmr::string>(gr::tag::RESET_DEFAULTS))) {
                resetDefaults();
            }

            // Use static dispatch table for O(1) lookup instead of O(members) iteration (Optimization F)
            const auto&  appliers = stagedAppliers();
            property_map staged;
            for (const auto& [key, stagedValue] : _stagedParameters) {
                auto it = appliers.find(key);
                if (it != appliers.end()) {
                    constexpr bool hasCallback = HasSettingsChangedCallback<TBlock>;
                    std::ignore                = it->second(_block, key, stagedValue, result.appliedParameters, staged, hasCallback);
                    // Forward parameters check is independent of validation success (matches original behavior)
                    if (_autoForwardParameters.contains(std::string(key))) {
                        result.forwardParameters.insert_or_assign(key, stagedValue);
                    }
                }
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
                    const float newSampleRate = ratio * (*_activeParameters.at(gr::tag::SAMPLE_RATE.shortKey()).template get_if<float>());
                    result.forwardParameters.insert_or_assign(gr::tag::SAMPLE_RATE.shortKey(), newSampleRate);
                }
            }

            if (_stagedParameters.contains(static_cast<std::pmr::string>(gr::tag::STORE_DEFAULTS))) {
                storeDefaults();
            }

            if constexpr (HasSettingsResetCallback<TBlock>) {
                if (_stagedParameters.contains(static_cast<std::pmr::string>(gr::tag::RESET_DEFAULTS))) {
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
        // Use static dispatch table for O(1) membership check instead of O(members) iteration (Optimization F)
        const auto&  setters = parameterSetters();
        property_map newProperties;

        for (const auto& [key, value] : parameters) {
            if (setters.contains(key)) {
                newProperties[key] = value;
            } else {
                auto str = ctx.context.value_or(std::string_view{});
                if (str.empty()) { // store meta_information only for default
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
        // Use static dispatch table for reduced template instantiation (Optimization F)
        const auto& readers = activeParameterReaders();
        for (const auto& reader : readers) {
            reader(_block, _activeParameters);
        }
    }

    NO_INLINE void storeCurrentParameters(property_map& parameters) {
        // Use static dispatch table for reduced template instantiation (Optimization F)
        if constexpr (refl::reflectable<TBlock>) {
            const auto& readers = parameterReaders();
            for (const auto& reader : readers) {
                reader(_block, parameters);
            }
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
