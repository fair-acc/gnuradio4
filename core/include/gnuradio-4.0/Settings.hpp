#ifndef GNURADIO_SETTINGS_HPP
#define GNURADIO_SETTINGS_HPP

#include <array>
#include <chrono>
#include <concepts>
#include <format>
#include <mutex>
#include <optional>
#include <set>
#include <variant>

#include <gnuradio-4.0/AtomicRef.hpp>
#include <gnuradio-4.0/BlockTraits.hpp>
#include <gnuradio-4.0/Complex.hpp>
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
    if constexpr (gr::meta::vector_type<T> || gr::meta::array_type<T> || is_tensor<T>) {
        using ValueType = typename T::value_type;
        // TODO(follow-up PR): remove Value as collection element — it bypasses C++ type safety and breaks settings introspection
        return std::is_arithmetic_v<ValueType> || std::is_same_v<ValueType, std::string> || std::is_same_v<ValueType, std::pmr::string> || std::is_same_v<ValueType, std::complex<double>> || std::is_same_v<ValueType, std::complex<float>> || std::is_same_v<ValueType, gr::complex<double>> || std::is_same_v<ValueType, gr::complex<float>> || std::is_enum_v<ValueType> || std::is_same_v<ValueType, Value>;
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
    // TODO(follow-up PR): remove Value as settings type — it erases type information, prevents validation, and complicates GRC YAML serialisation
    return std::is_arithmetic_v<T> || std::is_same_v<T, std::string> || std::is_same_v<T, std::pmr::string> || isSupportedVectorOrTensorType<T>() || std::is_same_v<T, property_map> //
           || std::is_same_v<T, std::complex<double>> || std::is_same_v<T, std::complex<float>> || std::is_same_v<T, gr::complex<double>> || std::is_same_v<T, gr::complex<float>> || std::is_enum_v<T> || std::is_same_v<T, Value> || isReadableImmutable();
}

template<typename T, typename TMember>
constexpr bool isWritableMember() {
    return isReadableMember<T>() && !std::is_const_v<T> && !std::is_const_v<TMember> && !gr::meta::is_immutable<TMember>{};
}

/// `CORE_NAMING_GUIDELINE.md` reserves a leading underscore for non-public fields. Such a member may still be
/// reflected -- that is what carries it with the block -- but it is off the public settings surface: it is neither
/// settable nor reported among the block's parameters, and `resetDefaults()` does not restore it.
[[nodiscard]] constexpr bool isPublicSettingName(std::string_view name) noexcept { return !name.starts_with('_'); }

/// the single test every walk over reflected members applies, so that no walk can forget it
template<typename TBlock, auto kIdx>
inline constexpr bool kIsPublicSetting = isPublicSettingName(refl::data_member_name<TBlock, kIdx>.view());

inline constexpr uint64_t convertTimePointToUint64Ns(const std::chrono::time_point<std::chrono::system_clock>& tp) {
    const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(tp.time_since_epoch()).count();
    return static_cast<uint64_t>(ns);
}

static auto nullMatchPred = [](auto, auto, auto) { return std::nullopt; };

} // namespace settings

struct ApplyStagedParametersResult {
    property_map forwardParameters; // parameters that should be forwarded to dependent child blocks
    property_map appliedParameters;
    property_map refusedParameters; // staged values the block did not take: outside their `Limits<>` or of the wrong type
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

std::size_t computeHash(const ValueView& value);

template<class T>
constexpr std::size_t hash_combine(std::size_t seed, const T& v) noexcept {
    std::hash<T> hasher;
    seed ^= hasher(v) + 0x9e3779b9UZ + (seed << 6) + (seed >> 2);
    return seed;
}

inline const auto computeValueHash = meta::overloaded([](const std::string_view& sv) { return std::hash<std::string_view>()(sv); }, //
    []<typename T>(const gr::Tensor<T>& tensor) {
        std::size_t seed = 9UZ;
        for (const auto& v : tensor) {
            seed = detail::hash_combine(seed, computeHash(Value(v)));
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

inline std::size_t computeHash(const ValueView& value) {
    std::size_t result = 0UZ;
    pmt::ValueVisitor([&result](const auto& v) { result = computeValueHash(v); }).visit(value);
    return result;
}

template<typename TCollection>
auto collectionToTensor(const TCollection& collection) {
    using TValue       = typename TCollection::value_type;
    using TTensorValue = std::conditional_t<std::is_same_v<std::string, TValue> || std::is_same_v<std::pmr::string, TValue>, Value, TValue>;
    Tensor<TTensorValue> result(extents_from, {collection.size()});
    std::ranges::copy(collection, result.begin());
    return result;
}

template<typename T, typename U = unwrap_if_wrapped_t<std::remove_cvref_t<T>>>
constexpr bool isEnumOrAnnotatedEnum = std::is_enum_v<U>;

template<typename T>
requires isEnumOrAnnotatedEnum<T>
std::expected<T, std::string> tryExtractEnumValue(const ValueView& pmt, std::string_view key) {

    auto str = pmt.value_or(std::string_view{});
    if (str.data() == nullptr) {
        return std::unexpected(std::format("Field '{}' expects enum string, got different type", key));
    }

    if (auto opt = gr::meta::parseEnum<T>(str); opt.has_value()) {
        return *opt;
    }

    return std::unexpected(std::format("Invalid enum value '{}' for key '{}'", str, key));
}

template<typename T, typename U = std::remove_cvref_t<T>>
requires isEnumOrAnnotatedEnum<U>
constexpr std::string_view enumToString(T&& enum_value) noexcept {
    // enumName returns a view into static reflection data; the map insert copies it into the blob.
    if constexpr (is_annotated<U>()) {
        return gr::meta::enumName(enum_value.value).value_or("");
    } else {
        return gr::meta::enumName(enum_value).value_or("");
    }
}

} // namespace detail

struct SettingsCtx {
    std::uint64_t time    = 0ULL; // UTC-based time-stamp in ns, time from which the setting is valid, 0U is undefined time
    std::string   context = {};   // user-defined multiplexing context for which the setting is valid

    auto operator<=>(const SettingsCtx&) const = default;

    [[nodiscard]] std::size_t hash() const noexcept { return detail::hash_combine(std::hash<std::uint64_t>()(time), context); }
};

/**
 * @brief a concept verifying whether a processing block optionally provides a `settingsChanged` callback to react to
 * block configuration changes and/or to influence forwarded downstream parameters.
 *
 * Implementers may have:
 * 1. `settingsChanged(oldSettings, newSettings)`
 * 2. `settingsChanged(oldSettings, newSettings, forwardSettings)`
 *    - `forwardSettings` influences subsequent blocks.
 *      A forwarded `sample_rate` is the block's input rate. Block converts it when publishing an output tag.
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
[[nodiscard]] std::expected<T, std::string> convertParameter(std::string_view key, const ValueView& value) {
    if constexpr (std::is_enum_v<T>) {
        return detail::tryExtractEnumValue<T>(value, key);
    } else {
        if constexpr (std::is_same_v<T, std::string> || std::is_same_v<T, std::pmr::string>) {
            auto sv = value.value_or(std::string_view{});
            if (sv.data()) {
                return T(sv);
            } else {
                return std::unexpected(std::format("value {} for key '{}' has wrong type {} {}, needs {}", value, key, value.value_type(), value.container_type(), std::string(meta::type_name<T>())));
            }

        } else if constexpr (meta::array_or_vector_type<T>) {
            using TValue      = typename T::value_type;
            using TTensorElem = std::conditional_t<std::is_same_v<TValue, std::string> || std::is_same_v<TValue, std::pmr::string>, Value, TValue>;
            Value                                  materialised; // owns the bytes tensorValue aliases (Tensor<Value> branch); declared here so the view outlives the if-constexpr block
            std::optional<TensorView<TTensorElem>> tensorValue;
            if constexpr (std::is_same_v<TTensorElem, Value>) {
                materialised = value.owned();
                tensorValue  = materialised.get_if<TensorView<TTensorElem>>();
            } else {
                tensorValue = value.template get_if<TensorView<TTensorElem>>();
            }
            if (!tensorValue) {
                return std::unexpected(std::format("Value {} is not a tensor of {}", value, meta::type_name<TTensorElem>()));
            }
            T converted;
            if constexpr (meta::array_type<T>) {
                if (tensorValue->size() != std::tuple_size_v<T>) {
                    return std::unexpected(std::format("tensor size {} does not match array size {}", tensorValue->size(), std::tuple_size_v<T>));
                }
            } else {
                converted.resize(tensorValue->size());
            }
            if constexpr (std::is_same_v<TValue, std::string> || std::is_same_v<TValue, std::pmr::string>) {
                std::ranges::transform(*tensorValue, converted.begin(), [](const Value& in) { return TValue(in.value_or(std::string_view{})); });
            } else {
                std::ranges::copy(*tensorValue, converted.begin());
            }
            return converted;

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

namespace detail {
// Free templates (not CtxSettings<TBlock>::) so the bodies instantiate once per Type
// instead of once per (TBlock, Type).
template<typename Type>
inline std::optional<std::string> setParameterImpl(std::string_view key, const ValueView& value, property_map& newParameters) {
    if (auto convertedValue = settings::convertParameter<Type>(key, value); convertedValue) [[likely]] {
        if constexpr (detail::isEnumOrAnnotatedEnum<Type>) {
            newParameters.insert_or_assign(key, detail::enumToString(convertedValue.value()));
        } else if constexpr (meta::array_or_vector_type<Type>) {
            newParameters.insert_or_assign(key, Value(detail::collectionToTensor(*convertedValue)));
        } else {
            newParameters.insert_or_assign(key, detail::castToGrSizeIfNeeded(convertedValue.value()));
        }
        return std::nullopt;
    } else {
        return convertedValue.error();
    }
}
} // namespace detail

/** one bit per public writable setting, by its row in the block's name-sorted settings table: set = the setting follows tags */
using AutoUpdateBits = std::pmr::vector<bool>;

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
     * a changed standard setting (`gr::tag::kDefaultTags`) is published downstream as a tag; other settings stay local
     */
    virtual void autoUpdate(const property_map_view& tagMap) = 0;

    /**
     * @brief return all (or for selected multiple keys) available active block settings as key-value pairs
     */
    [[nodiscard]] virtual property_map get(std::span<const std::string> parameter_keys = {}) const noexcept = 0;

    /**
     * @brief return available active block setting as key-value pair for a single key
     */
    [[nodiscard]] virtual std::optional<Value> get(const std::string& parameter_key) const noexcept = 0;

    /**
     * @brief return all (or for selected multiple keys) stored block settings for provided context as key-value pairs
     */
    [[nodiscard]] virtual std::optional<property_map> getStored(std::span<const std::string> parameterKeys = {}, SettingsCtx ctx = {}) const noexcept = 0;

    /**
     * @brief return available stored block setting for provided context as key-value pair for a single key
     */
    [[nodiscard]] virtual std::optional<Value> getStored(const std::string& parameter_key, SettingsCtx ctx = {}) const noexcept = 0;

    /**
     * @brief return number of all sets of stored parameters
     */
    [[nodiscard]] virtual gr::Size_t getNStoredParameters() const noexcept = 0;

    /**
     * @brief return number of sets of auto update parameters
     */
    [[nodiscard]] virtual gr::Size_t getNAutoUpdateParameters() const noexcept = 0;

    // transparent string-view comparator: cross-allocator lookups (std::string ↔ std::pmr::string ↔ string_view).
    struct PmrStringLess {
        using is_transparent = void;
        bool operator()(std::string_view a, std::string_view b) const noexcept { return a < b; }
    };
    using StoredParametersMap = std::pmr::map<std::pmr::string, std::vector<CtxSettingsPair>, PmrStringLess>;

    /**
     * @brief return _storedParameters
     */
    [[nodiscard]] virtual StoredParametersMap getStoredAll() const noexcept = 0;

    /**
     * @brief returns the staged/not-yet-applied new parameters
     */
    [[nodiscard]] virtual const property_map& stagedParameters() const = 0;

    /** the names of the settings that follow tags, sorted; a copy taken under the settings mutex */
    [[nodiscard]] virtual std::pmr::vector<std::string_view> autoUpdateParameters(SettingsCtx ctx = {}) noexcept = 0;

    [[nodiscard]] virtual const property_map& defaultParameters() const noexcept = 0;

    /** unlocked: the block's work thread writes it (applyStagedParameters, updateActiveParameters) and is the only thread that may read it by reference; others use get() */
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

    virtual void deferNotification(const property_map& appliedParameters) = 0;

    [[nodiscard]] virtual std::optional<property_map> takeDeferredNotification() = 0;

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
    using MatchPredicate = std::function<std::optional<bool>(std::string_view, std::string_view, std::size_t)>;

protected:
    mutable bool       _changed{false};
    mutable std::mutex _mutex{};

    // all routed through the block's mechanics resource (captured via TLS at construction)
    mutable StoredParametersMap                _storedParameters{std::pmr::polymorphic_allocator<>{ResourceProfile::currentTls().mechanicsResource()}};
    property_map                               _defaultParameters{ResourceProfile::currentTls().mechanicsResource()};
    property_map                               _initBlockParameters{ResourceProfile::currentTls().mechanicsResource()};
    std::pmr::map<SettingsCtx, AutoUpdateBits> _autoUpdateParameters{std::pmr::polymorphic_allocator<>{ResourceProfile::currentTls().mechanicsResource()}};
    MatchPredicate                             _matchPred = settings::nullMatchPred;
    SettingsCtx                                _activeCtx{};
    property_map                               _stagedParameters{ResourceProfile::currentTls().mechanicsResource()};
    property_map                               _activeParameters{ResourceProfile::currentTls().mechanicsResource()};
    mutable bool                               _notificationPending{false};
    property_map                               _deferredNotification{ResourceProfile::currentTls().mechanicsResource()};

    const std::size_t _timePrecisionTolerance = 100; // ns, now used for emscripten

    // Virtual hooks for type-dependent logic called from type-independent methods
    [[nodiscard]] virtual property_map                      doSetStagedImpl(const property_map& parameters) = 0;
    [[nodiscard]] virtual std::span<const std::string_view> doGetAllWritableMembers() const                 = 0;

public:
    // Settings configuration
    std::uint64_t expiry_time{std::numeric_limits<std::uint64_t>::max()};

    // --- Type-independent virtual method implementations (defined in Settings.cpp) ---

    [[nodiscard]] bool changed() const noexcept override;
    void               setChanged(bool b) noexcept override;
    void               setInitBlockParameters(const property_map& parameters) override;

    [[nodiscard]] const SettingsCtx& activeContext() const noexcept override;

    [[nodiscard]] const property_map& defaultParameters() const noexcept override;
    [[nodiscard]] const property_map& activeParameters() const noexcept override;

    [[nodiscard]] property_map         get(std::span<const std::string> parameterKeys = {}) const noexcept override;
    [[nodiscard]] std::optional<Value> get(const std::string& parameterKey) const noexcept override;

    [[nodiscard]] std::optional<property_map> getStored(std::span<const std::string> parameterKeys = {}, SettingsCtx ctx = {}) const noexcept override;
    [[nodiscard]] std::optional<Value>        getStored(const std::string& parameterKey, SettingsCtx ctx = {}) const noexcept override;

    [[nodiscard]] gr::Size_t getNStoredParameters() const noexcept override;
    [[nodiscard]] gr::Size_t getNAutoUpdateParameters() const noexcept override;

    [[nodiscard]] StoredParametersMap getStoredAll() const noexcept override;

    [[nodiscard]] const property_map& stagedParameters() const override;

    [[nodiscard]] std::pmr::vector<std::string_view> autoUpdateParameters(SettingsCtx ctx = {}) noexcept override;

    [[nodiscard]] property_map setStaged(const property_map& parameters) override;

    void                                      deferNotification(const property_map& appliedParameters) override;
    [[nodiscard]] std::optional<property_map> takeDeferredNotification() override;

    [[nodiscard]] std::optional<SettingsCtx> activateContext(SettingsCtx ctx = {}) override;
    [[nodiscard]] bool                       removeContext(SettingsCtx ctx) override;

    void assignFrom(const CtxSettingsBase& other);
    void assignFrom(CtxSettingsBase&& other) noexcept;

protected:
    // --- Private helpers (defined in Settings.cpp) ---
    [[nodiscard]] std::optional<std::string>    findBestMatchCtx(std::string_view contextToSearch) const;
    [[nodiscard]] std::optional<SettingsCtx>    findBestMatchSettingsCtx(const SettingsCtx& ctx) const;
    [[nodiscard]] std::optional<property_map>   getBestMatchStoredParameters(const SettingsCtx& ctx) const;
    [[nodiscard]] std::optional<AutoUpdateBits> getBestMatchAutoUpdateParameters(const SettingsCtx& ctx) const;
    [[nodiscard]] AutoUpdateBits                allWritableBits() const;
    void                                        resolveDuplicateTimestamp(SettingsCtx& ctx);
    void                                        addStoredParameters(const property_map& newParameters, const SettingsCtx& ctx);
    void                                        removeExpiredStoredParameters();
    void                                        amendStoredParameters(const property_map& staged, const property_map& kept); // a stored value equal to a refused staged one becomes the kept one
    [[nodiscard]] std::optional<std::string>    contextInTag(const property_map_view& tagMap) const;
    [[nodiscard]] std::optional<std::uint64_t>  contextTimeInTag(const property_map_view& tagMap) const;
    [[nodiscard]] std::optional<SettingsCtx>    createSettingsCtxFromTag(const property_map_view& tagMap) const;
}; // class CtxSettingsBase

template<typename TBlock>
class CtxSettings : public CtxSettingsBase {
    TBlock* _block = nullptr;

    // Virtual hook: delegates to type-dependent setStagedImpl using static dispatch table
    [[nodiscard]] property_map doSetStagedImpl(const property_map& parameters) override {
        property_map ret;
        if constexpr (refl::reflectable<TBlock>) {
            for (const auto& [key, value] : parameters) {
                const auto fieldKey = gr::tag::settingsKey(key);
                if (const SettingRow* row = settingRow(fieldKey)) {
                    if (auto error = row->set(fieldKey, value, _stagedParameters)) {
                        gr::log::warning(*error);
                        ret.insert_or_assign(fieldKey, value);
                    }
                } else if (isDefaultsControl(fieldKey)) {
                    stageDefaultsControl(fieldKey, value, _stagedParameters);
                } else {
                    ret.insert_or_assign(fieldKey, value);
                }
            }
        }
        if (!_stagedParameters.empty()) {
            setChanged(true);
        }
        return ret;
    }

    [[nodiscard]] std::span<const std::string_view> doGetAllWritableMembers() const override { return settingNames(); }

public:
    // Type aliases for dispatch function pointers
    using ParameterSetter       = std::optional<std::string> (*)(std::string_view key, const ValueView& value, property_map& newParameters);
    using StagedParameterSetter = std::optional<std::string> (*)(std::string_view key, const ValueView& value, property_map& stagedParameters);
    using StagedApplier         = bool (*)(TBlock* block, std::string_view key, const ValueView& value, property_map& applied, property_map& staged, bool hasCallback);
    using ParameterReader       = void (*)(const TBlock* block, property_map& parameters);
    using ActiveParameterReader = void (*)(const TBlock* block, property_map& activeParameters);

private:
    // Helper template for applyStagedParameters - applies value to block member
    template<std::size_t kIdx, typename RawType, typename Type>
    static bool applyStagedImpl(TBlock* block, std::string_view key, const ValueView& stagedValue, property_map& applied, property_map& staged, bool hasCallback) {
        auto& member = refl::data_member<kIdx>(*block);

        std::expected<Type, std::string> maybe_value;
        if constexpr (detail::isEnumOrAnnotatedEnum<RawType>) {
            maybe_value = detail::tryExtractEnumValue<Type>(stagedValue, key);
        } else if constexpr (std::is_same_v<Type, std::string> || std::is_same_v<Type, std::pmr::string>) {
            auto str = stagedValue.value_or(std::string_view{});
            if (str.data() != nullptr) {
                maybe_value = Type(str);
            } else {
                maybe_value = std::unexpected("Unexpected type in stagedValue");
            }
        } else if constexpr (meta::array_or_vector_type<Type>) {
            // convertParameter handles the TensorView validity check + (for Tensor<Value>) the
            // owning materialisation internally, returning std::unexpected with a descriptive
            // diagnostic on failure.
            maybe_value = settings::convertParameter<Type>(key, stagedValue);
#ifdef __EMSCRIPTEN__
        } else if constexpr (std::is_same_v<Type, std::size_t> && !std::is_same_v<std::size_t, gr::Size_t>) {
            const auto* ptr = stagedValue.get_if<gr::Size_t>(); // raw ptr (checked_access_ptr terminates on null)
            if (ptr != nullptr) {
                maybe_value = static_cast<std::size_t>(*ptr);
            } else {
                maybe_value = std::unexpected("Unexpected type in stagedValue");
            }
#endif
        } else if constexpr (std::is_same_v<Type, property_map>) {
            if (auto opt = stagedValue.template get_if<property_map>()) {
                maybe_value = opt->owned();
            } else {
                maybe_value = std::unexpected("Unexpected type in stagedValue");
            }
        } else if constexpr (gr::TensorLike<Type>) {
            // Appendix B: owning materialisation goes through value_or<Tensor<T>>(default), which
            // routes internally through get_if<TensorView<T>>().owned(_resource).
            using ElemT = std::remove_const_t<typename Type::value_type>;
            if constexpr (std::same_as<ElemT, pmt::Value>) {
                const pmt::Value owned(stagedValue);
                if (owned.template get_if<gr::TensorView<ElemT>>().has_value()) {
                    maybe_value = owned.template value_or<Type>(Type{});
                } else {
                    maybe_value = std::unexpected("Unexpected type in stagedValue");
                }
            } else if (stagedValue.template get_if<gr::TensorView<ElemT>>().has_value()) {
                maybe_value = stagedValue.template value_or<Type>(Type{});
            } else {
                maybe_value = std::unexpected("Unexpected type in stagedValue");
            }
        } else {
            const auto* ptr = stagedValue.get_if<Type>(); // raw ptr (checked_access_ptr terminates on null)
            if (ptr != nullptr) {
                maybe_value = *ptr;
            } else {
                maybe_value = std::unexpected("Unexpected type in stagedValue");
            }
        }

        if constexpr (is_annotated<RawType>()) {
            if (maybe_value && member.validate_and_set(*maybe_value)) {
                applied.insert_or_assign(key, stagedValue);
                if (hasCallback) {
                    staged.insert_or_assign(key, stagedValue);
                }
                return true;
            }
            return false;
        } else {
            if (!maybe_value) {
                return false;
            }
            member = *maybe_value;
            applied.insert_or_assign(key, stagedValue);
            if (hasCallback) {
                staged.insert_or_assign(key, stagedValue);
            }
            return true;
        }
    }

    // Helper template for storeCurrentParameters - reads member value into property_map
    template<std::size_t kIdx, typename Type>
    static void storeParameterImpl(const TBlock* block, property_map& parameters) {
        constexpr std::string_view key    = refl::data_member_name<TBlock, kIdx>.view();
        const auto&                member = refl::data_member<kIdx>(*block);
        if constexpr (detail::isEnumOrAnnotatedEnum<Type>) {
            parameters.insert_or_assign(key, detail::enumToString(member));
        } else if constexpr (meta::array_or_vector_type<Type>) {
            const auto& from = detail::unwrap_decorated_value(member);
            parameters.insert_or_assign(key, detail::collectionToTensor(from));
        } else {
            parameters.insert_or_assign(key, detail::unwrap_decorated_value(member));
        }
    }

    // Helper template for updateActiveParameters - reads member value for active parameters
    template<std::size_t kIdx, typename RawType, typename Type>
    static void updateActiveParameterImpl(const TBlock* block, property_map& activeParameters) {
        constexpr std::string_view key    = refl::data_member_name<TBlock, kIdx>.view();
        const auto&                member = refl::data_member<kIdx>(*block);
        if constexpr (detail::isEnumOrAnnotatedEnum<RawType>) {
            activeParameters.insert_or_assign(key, detail::enumToString(member));
        } else if constexpr (meta::array_or_vector_type<Type>) {
            const auto& from = detail::unwrap_decorated_reference(member);
            activeParameters.insert_or_assign(key, Value(detail::collectionToTensor(from)));
        } else {
            activeParameters.insert_or_assign(key, detail::unwrap_decorated_value(member));
        }
    }

public:
    struct SettingRow {
        std::string_view name;
        ParameterSetter  set;
        StagedApplier    applyStaged;
    };

    template<std::size_t kIdx>
    static constexpr bool kIsSettingMember = [] {
        using MemberType = refl::data_member_type<TBlock, kIdx>;
        using Type       = unwrap_if_wrapped_t<std::remove_cvref_t<MemberType>>;
        return settings::isWritableMember<Type, MemberType>() && settings::kIsPublicSetting<TBlock, kIdx>;
    }();

    [[nodiscard]] static consteval auto makeSettingRows() {
        constexpr std::size_t nRows = [] {
            std::size_t count = 0UZ;
            if constexpr (refl::reflectable<TBlock>) {
                refl::for_each_data_member_index<TBlock>([&count](auto kIdx) { count += kIsSettingMember<decltype(kIdx)::value> ? 1UZ : 0UZ; });
            }
            return count;
        }();
        std::array<SettingRow, nRows> rows{};
        if constexpr (refl::reflectable<TBlock>) {
            std::size_t row = 0UZ;
            refl::for_each_data_member_index<TBlock>([&rows, &row](auto kIdx) {
                constexpr std::size_t kI = decltype(kIdx)::value;
                using RawType            = std::remove_cvref_t<refl::data_member_type<TBlock, kI>>;
                using Type               = unwrap_if_wrapped_t<RawType>;
                if constexpr (kIsSettingMember<kI>) {
                    static_assert(tag::hasCanonicalSettingType<Type>(refl::data_member_name<TBlock, kI>.view()), "a setting named like a canonical tag key must have the tag's value type, or a std::vector of it");
                    rows[row++] = SettingRow{refl::data_member_name<TBlock, kI>.view(), &detail::setParameterImpl<Type>, &applyStagedImpl<kI, RawType, Type>};
                }
            });
        }
        std::ranges::sort(rows, {}, &SettingRow::name);
        return rows;
    }

    [[nodiscard]] static std::span<const SettingRow> settingRows() noexcept {
        static constexpr auto kRows = makeSettingRows();
        return kRows;
    }

    [[nodiscard]] static std::span<const std::string_view> settingNames() noexcept {
        static constexpr auto kNames = [] {
            constexpr auto                             kRows = makeSettingRows();
            std::array<std::string_view, kRows.size()> result{};
            std::ranges::transform(kRows, result.begin(), &SettingRow::name);
            return result;
        }();
        return kNames;
    }

    [[nodiscard]] static const SettingRow* settingRow(std::string_view name) noexcept {
        const auto rows = settingRows();
        const auto it   = std::ranges::lower_bound(rows, name, {}, &SettingRow::name);
        return it != rows.end() && it->name == name ? &*it : nullptr;
    }

    [[nodiscard]] static bool isSetting(std::string_view name) noexcept { return settingRow(name) != nullptr; }

    [[nodiscard]] static bool isDefaultsControl(std::string_view name) noexcept { return name == gr::tag::RESET_DEFAULTS.shortKey() || name == gr::tag::STORE_DEFAULTS.shortKey(); }

    static void stageDefaultsControl(std::string_view name, const ValueView& value, property_map& staged) {
        if (pmt::convert_safely<bool>(value).value_or(false)) {
            staged.insert_or_assign(name, pmt::Value(true));
        }
    }

    [[nodiscard]] static std::size_t rowIndex(const SettingRow& row) noexcept { return static_cast<std::size_t>(&row - settingRows().data()); }

    [[nodiscard]] static const std::array<bool, gr::pmt::keys::kCanonical.size() + 1UZ>& canonicalSettingIds() {
        static const auto ids = gr::pmt::keys::idFlagsWhere([](std::string_view name) { return isSetting(name); });
        return ids;
    }

    [[nodiscard]] static const std::array<bool, gr::pmt::keys::kCanonical.size() + 1UZ>& canonicalRelevantIds() {
        static const auto ids = gr::pmt::keys::idFlagsWhere([](std::string_view name) { return isSetting(name) || isDefaultsControl(name) || name == gr::tag::CONTEXT.shortKey(); });
        return ids;
    }

    // Static list of parameter readers for storeCurrentParameters()
    [[nodiscard]] static const std::vector<ParameterReader>& parameterReaders() {
        static const std::vector<ParameterReader> readers = [] {
            std::vector<ParameterReader> result;
            if constexpr (refl::reflectable<TBlock>) {
                refl::for_each_data_member_index<TBlock>([&result](auto kIdx) {
                    using MemberType = refl::data_member_type<TBlock, kIdx>;
                    using Type       = unwrap_if_wrapped_t<std::remove_cvref_t<MemberType>>;
                    if constexpr (settings::isReadableMember<Type>() && settings::kIsPublicSetting<TBlock, kIdx>) {
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
                refl::for_each_data_member_index<TBlock>([&result](auto kIdx) {
                    using MemberType = refl::data_member_type<TBlock, kIdx>;
                    using RawType    = std::remove_cvref_t<MemberType>;
                    using Type       = unwrap_if_wrapped_t<RawType>;
                    if constexpr (settings::isReadableMember<Type>() && settings::kIsPublicSetting<TBlock, kIdx>) {
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
                _block->meta_information.value.insert_or_assign(std::string_view{"description"}, std::string(_block->description));
            }

            // handle meta-information for UI and other non-processing-related purposes
            refl::for_each_data_member_index<TBlock>([this](auto kIdx) {
                using MemberType = refl::data_member_type<TBlock, kIdx>;
                using RawType    = std::remove_cvref_t<MemberType>;
                using Type       = unwrap_if_wrapped_t<RawType>;

                if constexpr (hasMetaInfo && std::is_enum_v<Type> && settings::kIsPublicSetting<TBlock, kIdx>) {
                    auto  memberName = std::string(refl::data_member_name<TBlock, kIdx>.view());
                    auto& meta_info  = _block->meta_information.value;
                    meta_info.insert_or_assign(convert_string_domain(memberName) + "::enum_values", Value{[] {
                        constexpr auto           values = gr::meta::enumValues<Type>();
                        std::vector<std::string> result;
                        result.reserve(values.size());
                        for (auto v : values) {
                            if (auto name = gr::meta::enumName(v); name.has_value()) {
                                result.emplace_back(*name);
                            }
                        }
                        return result;
                    }()});
                    meta_info.insert_or_assign(convert_string_domain(memberName) + "::enum_type", std::string(gr::meta::type_name<Type>()));
                }

                if constexpr (hasMetaInfo && AnnotatedType<RawType> && settings::kIsPublicSetting<TBlock, kIdx>) {
                    auto  memberName       = std::string(refl::data_member_name<TBlock, kIdx>.view());
                    auto& meta_information = _block->meta_information.value;
                    meta_information.insert_or_assign(convert_string_domain(memberName) + "::description", std::string(RawType::description()));
                    meta_information.insert_or_assign(convert_string_domain(memberName) + "::documentation", std::string(RawType::documentation()));
                    meta_information.insert_or_assign(convert_string_domain(memberName) + "::unit", std::string(RawType::unit()));
                    meta_information.insert_or_assign(convert_string_domain(memberName) + "::visible", RawType::visible());

                    using LimitType                    = typename RawType::LimitType;
                    using LimitValueType               = typename LimitType::ValueType;
                    constexpr bool kSpansNumericLimits = [kIdx] {
                        (void)kIdx;
                        if constexpr (std::numeric_limits<LimitValueType>::is_specialized) {
                            return LimitType::MinRange == std::numeric_limits<LimitValueType>::lowest() && LimitType::MaxRange == std::numeric_limits<LimitValueType>::max();
                        } else {
                            return false;
                        }
                    }();
                    if constexpr (LimitType::MinRange != LimitType::MaxRange && !kSpansNumericLimits) {
                        meta_information.insert_or_assign(convert_string_domain(memberName) + "::min", static_cast<Type>(LimitType::MinRange));
                        meta_information.insert_or_assign(convert_string_domain(memberName) + "::max", static_cast<Type>(LimitType::MaxRange));
                    }
                }
            });
        }

        storeDefaults();

        if (const property_map failed = set(_initBlockParameters); !failed.empty()) {
            gr::log::fatal("settings could not be applied: {}", failed);
        }

        if (const auto failed = activateContext(); failed == std::nullopt) {
            gr::log::fatal("Settings for context could not be activated");
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
                auto inherited             = getBestMatchAutoUpdateParameters(ctx);
                _autoUpdateParameters[ctx] = inherited ? std::move(*inherited) : allWritableBits();
            }
            auto& currentAutoUpdateParameters = _autoUpdateParameters[ctx];

            // Use static dispatch table for O(1) lookup instead of O(members) iteration (Optimization F)
            for (const auto& [key, value] : parameters) {
                if (value.is_monostate()) {
                    continue;
                }

                const auto fieldKey = gr::tag::settingsKey(key);
                if (const SettingRow* row = settingRow(fieldKey)) {
                    if (auto error = row->set(fieldKey, value, newParameters)) {
                        gr::log::warning(*error);
                        ret.insert_or_assign(fieldKey, value);
                        continue;
                    }
                    currentAutoUpdateParameters[rowIndex(*row)] = false;
                } else {
                    ret.insert_or_assign(fieldKey, value);
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

    NO_INLINE void autoUpdate(const property_map_view& tagMap) override {
        if constexpr (refl::reflectable<TBlock>) {
            // Fast path: tag carries no key that this block can act on. Skip the ctx-resolution
            // and per-key dispatch entirely; only the `setChanged(false)` bookkeeping remains
            // (and that itself is only required when no staged parameters are pending).
            auto tagCarriesRelevantKey = [&]() noexcept {
                if (tagMap.empty()) {
                    return false;
                }
                // recognise both the short ("sample_rate", "context") and the prefixed wire-format form ("gr:sample_rate", "gr:context")
                const auto& relevantIds = canonicalRelevantIds();
                const auto  keys        = tagMap.keys();
                for (auto it = keys.begin(); it != keys.end(); ++it) { // alloc-free key-only iter
                    const std::uint16_t id = it.keyId();
                    if (id != gr::pmt::keys::kIdUnknown && id < relevantIds.size()) { // canonical → O(1) relevance test
                        if (relevantIds[id]) {
                            return true;
                        }
                        continue;
                    }
                    const std::string_view fieldKey = gr::tag::settingsKey(*it);
                    if (isSetting(fieldKey) || isDefaultsControl(fieldKey) || fieldKey == gr::tag::CONTEXT.shortKey()) {
                        return true;
                    }
                }
                return false;
            };
            if (!tagCarriesRelevantKey()) {
                if (!gr::atomic_ref(_changed).load_acquire()) {
                    return;
                }
                std::lock_guard lg(_mutex);
                if (_stagedParameters.empty()) {
                    setChanged(false);
                }
                return;
            }

            std::lock_guard lg(_mutex);
            const auto      tagCtx      = createSettingsCtxFromTag(tagMap);
            const auto      previousCtx = _activeCtx; // capture before activateContext may change it

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

            const bool activeCtxChanged = previousCtx != ctx;

            // fuzzy-match auto-update parameters (exact lookup may fail due to timestamp mismatch)
            auto [paramIt, inserted] = _autoUpdateParameters.try_emplace(ctx);
            if (inserted) {
                auto inherited  = getBestMatchAutoUpdateParameters(ctx);
                paramIt->second = inherited ? std::move(*inherited) : allWritableBits();
            }
            const AutoUpdateBits& followsTag = paramIt->second;

            bool wasChanged = false;
            for (const auto& [key, value] : tagMap) {
                const std::string_view stripped = gr::tag::settingsKey(key);
                if (isDefaultsControl(stripped)) {
                    stageDefaultsControl(stripped, value, _stagedParameters);
                    wasChanged = true;
                    continue;
                }
                if (const SettingRow* row = settingRow(stripped); row != nullptr && followsTag[rowIndex(*row)]) {
                    if (!activeCtxChanged) {
                        if (auto active = _activeParameters.find(stripped); active != _activeParameters.end() && (*active).second == value) {
                            continue;
                        }
                    }
                    if (const auto error = row->set(stripped, value, _stagedParameters)) [[unlikely]] {
                        gr::log::warning("block '{}': tag value dropped: {}", blockName(), *error);
                        continue;
                    }
                    wasChanged = true;
                }
            }

            if (tagCtx == std::nullopt && !wasChanged && _stagedParameters.empty()) {
                setChanged(false);
            } else if (activeCtxChanged || wasChanged) {
                setChanged(true);
            }
        }
    }

    [[nodiscard]] NO_INLINE ApplyStagedParametersResult applyStagedParameters() override {
        ApplyStagedParametersResult result;
        if constexpr (refl::reflectable<TBlock>) {
            const auto [resetRequested, storeRequested] = [this] {
                std::lock_guard lg(_mutex);
                const bool      reset = _stagedParameters.erase(gr::tag::RESET_DEFAULTS.shortKey()) > 0UZ;
                const bool      store = _stagedParameters.erase(gr::tag::STORE_DEFAULTS.shortKey()) > 0UZ;
                return std::pair{reset, store};
            }();
            if (resetRequested) {
                resetDefaults();
                if (storeRequested) {
                    storeDefaults();
                }
                return result;
            }
            std::lock_guard lg(_mutex);

            // prepare old settings if required
            property_map oldSettings;
            if constexpr (HasSettingsChangedCallback<TBlock>) {
                storeCurrentParameters(oldSettings);
            }

            property_map staged;
            for (const auto& [key, stagedValue] : _stagedParameters) {
                if (const SettingRow* row = settingRow(key)) {
                    constexpr bool hasCallback = HasSettingsChangedCallback<TBlock>;
                    if (!row->applyStaged(_block, key, stagedValue, result.appliedParameters, staged, hasCallback)) {
                        result.refusedParameters.insert_or_assign(key, stagedValue);
                    }
                    // Forward parameters check is independent of validation success (matches original behavior)
                    if (gr::pmt::keys::lookupShortId(std::string_view{key}) != gr::pmt::keys::kIdUnknown) {
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

            property_map kept; // what the block holds instead of what was staged: refused, or put back by settingsChanged
            for (const auto& [key, stagedValue] : _stagedParameters) {
                if (const auto active = _activeParameters.find(key); active != _activeParameters.end() && !((*active).second == stagedValue)) {
                    kept.insert_or_assign(key, (*active).second);
                }
            }
            amendStoredParameters(_stagedParameters, kept);

            if (storeRequested) {
                storeDefaults();
            }
        }
        _stagedParameters.clear();
        gr::atomic_ref(_changed).store_release(false);
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
            if (isSetting(key)) {
                newProperties.insert_or_assign(key, value);
            } else if (ctx.context.empty()) { // store meta_information only for default
                _block->meta_information.value.insert_or_assign(key, value);
            }
        }

        if (const property_map failed = set(newProperties, ctx); !failed.empty()) {
            gr::log::fatal("settings from property_map could not be loaded: {}", failed);
        }
    }

private:
    [[nodiscard]] std::string_view blockName() const noexcept {
        if constexpr (requires { std::string_view{_block->name.value}; }) {
            return std::string_view{_block->name.value};
        } else {
            return gr::meta::type_name<TBlock>();
        }
    }

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
