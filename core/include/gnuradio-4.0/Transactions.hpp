#ifndef GNURADIO_TRANSACTIONS_HPP
#define GNURADIO_TRANSACTIONS_HPP

#include <atomic>
#include <cassert>
#include <chrono>
#include <concepts>
#include <functional>
#include <list>
#include <span>
#include <tuple>
#include <unordered_map>
#include <utility>

#include <pmtv/pmt.hpp>

#include <gnuradio-4.0/meta/formatter.hpp>
#include <gnuradio-4.0/meta/utils.hpp>

#include "PortTraits.hpp"
#include "Settings.hpp"
#include "Tag.hpp"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat-nonliteral"
#include <fmt/chrono.h>
#pragma GCC diagnostic pop

namespace gr {

static auto nullMatchPred = [](auto, auto, auto) { return std::nullopt; };

namespace settings {
/**
 * Policy for handling Settings Auto Update when `context` is not found in storedParameters.
 */
enum class AutoUpdatePolicy {
    Ignore,     // do nothing when context is not in storedParameters
    AddToStored // Add the new context to storedParameters with empty parameters and apply any new changes present in the Tag
};
} // namespace settings

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
    using MatchPredicate = std::function<std::optional<bool>(const pmtv::pmt &, const pmtv::pmt &, std::size_t)>;

    // pmtv::pmt comparison is needed to use it as a key of std::map
    struct PMTCompare {
        bool
        operator()(const pmtv::pmt &lhs, const pmtv::pmt &rhs) const {
            // If the types are different, cast rhs to the type of lhs and compare
            if (lhs.index() != rhs.index()) {
                // TODO: throw if types are not the same?
                return lhs.index() < rhs.index();
            } else {
                return std::visit(
                        [&](const auto &left) -> bool {
                            using T = std::decay_t<decltype(left)>;
                            if constexpr ((pmtv::String<T> || pmtv::Scalar<T>) && !pmtv::Complex<T>) {
                                return left < std::get<T>(rhs);
                            } else {
                                throw gr::exception("Invalid CtxSettings context type " + std::string(typeid(T).name()));
                            }
                            return false;
                        },
                        lhs);
            }
        }
    };

    TBlock            *_block = nullptr;
    mutable std::mutex _lock{};
    property_map       _activeParameters{};
    // key is SettingsCtx.context, value: queue of parameters with the same SettingsCtx.context but for different time
    mutable std::map<pmtv::pmt, std::vector<std::pair<SettingsCtx, property_map>>, PMTCompare> _storedParameters{};
    property_map                                                                               _defaultParameters{};
    std::set<std::string, std::less<>>                                                         _allWritableMembers{};   // all `isWritableMember` class members
    std::map<pmtv::pmt, std::set<std::string, std::less<>>, PMTCompare>                        _autoUpdateParameters{}; // for each SettingsCtx.context auto updated members are store separately
    std::set<std::string, std::less<>>                                                         _autoForwardParameters{};
    MatchPredicate                                                                             _matchPred        = nullMatchPred;
    pmtv::pmt                                                                                  _activeCtx        = "";
    settings::AutoUpdatePolicy                                                                 _autoUpdatePolicy = settings::AutoUpdatePolicy::AddToStored;

public:
    explicit CtxSettings(TBlock &block, MatchPredicate matchPred = nullMatchPred) noexcept : SettingsBase(), _block(&block), _matchPred(matchPred) {
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
            auto processOneMember = [&](auto member) {
                using RawType         = std::remove_cvref_t<decltype(member(*_block))>;
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
            processMembers<TBlock>(processOneMember);
        }
        addStoredParameters(property_map(), SettingsCtx(0ULL, ""));
    }

    constexpr CtxSettings(const CtxSettings &other) noexcept : SettingsBase(other) {
        CtxSettings temp(other);
        swap(temp);
    }

    constexpr CtxSettings(CtxSettings &&other) noexcept : SettingsBase(std::move(other)) {
        CtxSettings temp(std::move(other));
        swap(temp);
    }

    CtxSettings &
    operator=(const CtxSettings &other) noexcept {
        swap(other);
        return *this;
    }

    CtxSettings &
    operator=(CtxSettings &&other) noexcept {
        CtxSettings temp(std::move(other));
        swap(temp);
        return *this;
    }

    void
    swap(CtxSettings &other) noexcept {
        if (this == &other) {
            return;
        }
        SettingsBase::swap(other);
        std::swap(_block, other._block);
        std::scoped_lock lock(_lock, other._lock);
        std::swap(_activeParameters, other._activeParameters);
        std::swap(_storedParameters, other._storedParameters);
        std::swap(_defaultParameters, other._defaultParameters);
        std::swap(_allWritableMembers, other._allWritableMembers);
        std::swap(_autoUpdateParameters, other._autoUpdateParameters);
        std::swap(_autoForwardParameters, other._autoForwardParameters);
        std::swap(_matchPred, other._matchPred);
        std::swap(_activeCtx, other._activeCtx);
    }

    [[nodiscard]] property_map
    set(const property_map &parameters, SettingsCtx ctx = {}) override {
        property_map ret;
        if constexpr (refl::is_reflectable<TBlock>()) {
            std::lock_guard lg(_lock);
            if (ctx.time == 0ULL) {
                ctx.time = settings::convertTimePointToUint64Ns(std::chrono::system_clock::now());
            }
            property_map newParameters = getBestMatchStoredParameters(ctx);
            if (_autoUpdateParameters.find(ctx.context) == _autoUpdateParameters.end()) {
                _autoUpdateParameters[ctx.context] = _allWritableMembers;
            }

            for (const auto &[key, value] : parameters) {
                bool isSet            = false;
                auto processOneMember = [&, this](auto member) {
                    using Type = unwrap_if_wrapped_t<std::remove_cvref_t<decltype(member(*_block))>>;
                    if constexpr (settings::isWritableMember<Type>(member)) {
                        const auto fieldName = std::string_view(get_display_name(member));
                        if (fieldName == key && std::holds_alternative<Type>(value)) {
                            if (_autoUpdateParameters[ctx.context].contains(key)) {
                                _autoUpdateParameters[ctx.context].erase(key);
                            }
                            newParameters.insert_or_assign(key, value);
                            SettingsBase::_changed.store(true);
                            isSet = true;
                        }
                        if (fieldName == key && !std::holds_alternative<Type>(value)) {
                            throw std::invalid_argument([&key, &value] { // lazy evaluation
                                const std::size_t actual_index   = value.index();
                                const std::size_t required_index = meta::to_typelist<pmtv::pmt>::index_of<Type>(); // This too, as per your implementation.
                                return fmt::format("value for key '{}' has a wrong type. Index of actual type: {} ({}), Index of expected type: {} ({})", key, actual_index, "<missing pmt type>",
                                                   required_index, gr::meta::type_name<Type>());
                            }());
                        }
                    }
                };
                processMembers<TBlock>(processOneMember);
                if (!isSet) {
                    ret.insert_or_assign(key, pmtv::pmt(value));
                }
            }
            addStoredParameters(newParameters, ctx);
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

    void
    storeDefaults() override {
        this->storeDefaultSettings(_defaultParameters);
    }

    void
    resetDefaults() override {
        _storedParameters.clear();
        _autoUpdateParameters.clear();
        addStoredParameters(_defaultParameters, SettingsCtx(settings::convertTimePointToUint64Ns(std::chrono::system_clock::now()), ""));
        std::ignore = applyStagedParameters();
        if constexpr (HasSettingsResetCallback<TBlock>) {
            _block->reset();
        }
    }

    void
    autoUpdate(const Tag &tag) override {
        if constexpr (refl::is_reflectable<TBlock>()) {
            SettingsCtx ctx = createSettingsCtxFromTag(tag);

            auto bestMatchCtx = findBestMatchCtx(ctx.context);
            if (bestMatchCtx == std::nullopt) {
                if (_autoUpdatePolicy == settings::AutoUpdatePolicy::AddToStored) {
                    bestMatchCtx = ctx.context;
                } else if (_autoUpdatePolicy == settings::AutoUpdatePolicy::Ignore) {
                    return;
                }
            }

            const auto  found                = _autoUpdateParameters.find(bestMatchCtx.value());
            const auto &autoUpdateParameters = found == _autoUpdateParameters.end() ? _allWritableMembers : found->second;

            property_map        newParameters = getBestMatchStoredParameters(ctx);
            const property_map &parameters    = tag.map;
            for (const auto &[key, value] : parameters) {
                auto processOneMember = [&](auto member) {
                    using Type = unwrap_if_wrapped_t<std::remove_cvref_t<decltype(member(*_block))>>;
                    if constexpr (settings::isWritableMember<Type>(member)) {
                        if (autoUpdateParameters.contains(key) && std::string(get_display_name(member)) == key && std::holds_alternative<Type>(value)) {
                            newParameters.insert_or_assign(key, value);
                            SettingsBase::_changed.store(true);
                        }
                    }
                };
                processMembers<TBlock>(processOneMember);
            }
            addStoredParameters(newParameters, SettingsCtx(ctx.time, bestMatchCtx.value()));
            _activeCtx = bestMatchCtx.value();
        }
    }

    [[nodiscard]] const property_map
    stagedParameters(SettingsCtx ctx = {}) const noexcept override {
        std::lock_guard lg(_lock);
        return hasStoredParameters(ctx) ? getBestMatchStoredParameters(ctx) : property_map();
    }

    [[nodiscard]] property_map
    get(std::span<const std::string> parameterKeys = {}) const noexcept override {
        std::lock_guard lg(_lock);
        if (parameterKeys.empty()) {
            return _activeParameters;
        }
        property_map ret;
        for (const auto &key : parameterKeys) {
            if (_activeParameters.contains(key)) {
                ret.insert_or_assign(key, _activeParameters.at(key));
            }
        }
        return ret;
    }

    [[nodiscard]] std::optional<pmtv::pmt>
    get(const std::string &parameterKey) const noexcept override {
        auto res = get(std::array<std::string, 1>({ parameterKey }));
        if (res.contains(parameterKey)) {
            return res.at(parameterKey);
        } else {
            return std::nullopt;
        }
    }

    [[nodiscard]] property_map
    getStored(std::span<const std::string> parameterKeys = {}, SettingsCtx ctx = {}) const noexcept override {
        std::lock_guard     lg(_lock);
        const property_map &allBestMatchParameters = this->getBestMatchStoredParameters(ctx);

        if (parameterKeys.empty()) {
            return allBestMatchParameters;
        }
        property_map ret;
        for (const auto &key : parameterKeys) {
            if (allBestMatchParameters.contains(key)) {
                ret.insert_or_assign(key, allBestMatchParameters.at(key));
            }
        }
        return ret;
    }

    [[nodiscard]] std::optional<pmtv::pmt>
    getStored(const std::string &parameterKey, SettingsCtx ctx = {}) const noexcept override {
        auto res = getStored(std::array<std::string, 1>({ parameterKey }), ctx);
        if (res.contains(parameterKey)) {
            return res.at(parameterKey);
        } else {
            return std::nullopt;
        }
    }

    [[nodiscard]] std::set<std::string, std::less<>>
    autoUpdateParameters(SettingsCtx ctx = {}) noexcept override {
        auto bestMatchCtx = findBestMatchCtx(ctx.context);
        return bestMatchCtx == std::nullopt ? std::set<std::string, std::less<>>() : _autoUpdateParameters[bestMatchCtx.value()];
    }

    [[nodiscard]] std::set<std::string, std::less<>> &
    autoForwardParameters() noexcept override {
        return _autoForwardParameters;
    }

    [[nodiscard]] ApplyStagedParametersResult
    applyStagedParameters(std::uint64_t currentTime = 0ULL) noexcept override {
        ApplyStagedParametersResult result;
        if constexpr (refl::is_reflectable<TBlock>()) {
            std::lock_guard lg(_lock);

            // prepare old settings if required
            property_map oldSettings;
            if constexpr (HasSettingsChangedCallback<TBlock>) {
                storeDefaultSettings(oldSettings);
            }

            const SettingsCtx  activeSettingsCtx{ currentTime, _activeCtx };
            const property_map bestMatchParameters = getBestMatchStoredParameters(activeSettingsCtx);

            // check if reset of settings should be performed
            if (bestMatchParameters.contains(gr::tag::RESET_DEFAULTS)) {
                resetDefaults();
            }

            // update staged and forward parameters based on member properties
            property_map staged;
            for (const auto &[key, stagedValue] : bestMatchParameters) {
                auto applyOneMemberChanges = [&key, &staged, &result, &stagedValue, this](auto member) {
                    using RawType = std::remove_cvref_t<decltype(member(*_block))>;
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
                                               _block->unique_name, _block->name, member(*_block), std::get<Type>(stagedValue),                                   //
                                               std::string(get_display_name(member)), RawType::LimitType::MinRange,
                                               RawType::LimitType::MaxRange, //
                                               RawType::LimitType::ValidatorFunc == nullptr ? "not" : "");
#else
                                    fmt::print(stderr, " cannot set field {}({})::{} = {} to {} due to limit constraints [{}, {}] validate func is {} defined\n", //
                                               "_block->uniqueName", "_block->name", member(*_block), std::get<Type>(stagedValue),                                //
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
                processMembers<TBlock>(applyOneMemberChanges);
            }

            // update active parameters
            auto updateActive = [this](auto member) {
                using Type = unwrap_if_wrapped_t<std::remove_cvref_t<decltype(member(*_block))>>;
                if constexpr (settings::isReadableMember<Type>(member)) {
                    _activeParameters.insert_or_assign(get_display_name(member), static_cast<Type>(member(*_block)));
                }
            };
            processMembers<TBlock>(updateActive);

            // invoke user-callback function if staged is not empty
            if (!staged.empty()) {
                if constexpr (requires { _block->settingsChanged(/* old settings */ _activeParameters, /* new settings */ staged); }) {
                    _block->settingsChanged(/* old settings */ oldSettings, /* new settings */ staged);
                } else if constexpr (requires { _block->settingsChanged(/* old settings */ _activeParameters, /* new settings */ staged, /* new forward settings */ result.forwardParameters); }) {
                    _block->settingsChanged(/* old settings */ oldSettings, /* new settings */ staged, /* new forward settings */ result.forwardParameters);
                }
            }

            if (bestMatchParameters.contains(gr::tag::STORE_DEFAULTS)) {
                storeDefaults();
            }

            if constexpr (HasSettingsResetCallback<TBlock>) {
                if (bestMatchParameters.contains(gr::tag::RESET_DEFAULTS)) {
                    _block->reset();
                }
            }
            removeExpiredStoredParameters(activeSettingsCtx);
        }

        SettingsBase::_changed.store(false);
        return result;
    }

    void
    updateActiveParameters() noexcept override {
        if constexpr (refl::is_reflectable<TBlock>()) {
            std::lock_guard lg(_lock);
            auto            processOneMember = [&, this](auto member) {
                using Type = unwrap_if_wrapped_t<std::remove_cvref_t<decltype(member(*_block))>>;
                if constexpr (settings::isReadableMember<Type>(member)) {
                    _activeParameters.insert_or_assign(get_display_name_const(member).str(), member(*_block));
                }
            };
            processMembers<TBlock>(processOneMember);
        }
    }

private:
    [[nodiscard]] std::optional<pmtv::pmt>
    findBestMatchCtx(const pmtv::pmt &contextToSearch) const {
        if (_storedParameters.empty()) {
            return std::nullopt;
        }

        // exact match
        if (_storedParameters.find(contextToSearch) != _storedParameters.end()) {
            return contextToSearch;
        }

        // retry until we either get a match or std::nullopt
        for (std::size_t attempt = 0;; ++attempt) {
            for (const auto &i : _storedParameters) {
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

    [[nodiscard]] bool
    hasStoredParameters(const SettingsCtx &ctx) const {
        const auto bestMatchCtx = findBestMatchCtx(ctx.context);
        if (bestMatchCtx == std::nullopt) {
            return false;
        }
        return !_storedParameters[bestMatchCtx.value()].empty();
    }

    [[nodiscard]] property_map
    getBestMatchStoredParameters(const SettingsCtx &ctx) const {
        const auto bestMatchCtx = findBestMatchCtx(ctx.context);
        if (bestMatchCtx == std::nullopt) {
            return property_map();
        }
        const auto &vec = _storedParameters[bestMatchCtx.value()];
        if (vec.empty()) {
            return property_map();
        }
        if (ctx.time == 0ULL || vec.back().first.time <= ctx.time) {
            return vec.back().second;
        } else {
            auto lower = std::ranges::lower_bound(vec, ctx.time, {}, [](const auto &a) { return a.first.time; });
            if (lower == vec.end()) {
                return vec.back().second;
            } else {
                if (lower->first.time == ctx.time) {
                    return lower->second;
                } else if (lower != vec.begin()) {
                    --lower;
                    return lower->second;
                }
            }
        }
        return property_map();
    }

    void
    addStoredParameters(const property_map &newParameters, const SettingsCtx &ctx) {
        _storedParameters[ctx.context].push_back({ ctx, newParameters });
        auto &vec = _storedParameters[ctx.context];
        std::sort(vec.begin(), vec.end(), [](const auto &a, const auto &b) { return a.first.time < b.first.time; });

        if (_autoUpdateParameters.find(ctx.context) == _autoUpdateParameters.end()) {
            _autoUpdateParameters[ctx.context] = _allWritableMembers;
        }
    }

    void
    removeExpiredStoredParameters(const SettingsCtx &ctx) {
        const auto bestMatchCtx = findBestMatchCtx(ctx.context);
        if (bestMatchCtx == std::nullopt) {
            return;
        }
        auto &vec = _storedParameters[bestMatchCtx.value()];
        if (vec.empty()) {
            return;
        }
        if (ctx.time == 0ULL || vec.back().first.time <= ctx.time) {
            vec.clear();
        } else {
            auto lower = std::ranges::lower_bound(vec, ctx.time, {}, [](const auto &a) { return a.first.time; });
            if (lower == vec.end()) {
                vec.clear();
            } else {
                if (lower->first.time == ctx.time) {
                    vec.erase(vec.begin(), lower + 1);
                } else if (lower != vec.begin()) {
                    vec.erase(vec.begin(), lower);
                }
            }
        }
    }

    [[nodiscard]] SettingsCtx
    createSettingsCtxFromTag(const Tag &tag) const {
        // If TRIGGER_META_INFO is not present then context =_activeCtx, time = now()
        // If CONTEXT is not present then context =_activeCtx
        // IF TRIGGER_TIME is not present then time = now()

        const property_map &parameters = tag.map;
        SettingsCtx         ctx(0ULL, _activeCtx);

        // update if trigger_meta_info is present
        if (parameters.contains(gr::tag::TRIGGER_META_INFO.shortKey())) {
            const pmtv::pmt &pmtMetaInfo = parameters.at(std::string(gr::tag::TRIGGER_META_INFO.shortKey()));
            if (std::holds_alternative<property_map>(pmtMetaInfo)) {
                const property_map &metaInfo = std::get<property_map>(pmtMetaInfo);

                // update context if present
                if (metaInfo.contains(std::string(gr::tag::CONTEXT.shortKey()))) {
                    ctx.context = metaInfo.at(std::string(gr::tag::CONTEXT.shortKey()));
                }
            }
        }

        // update trigger time if present
        if (parameters.contains(gr::tag::TRIGGER_TIME.shortKey())) {
            const pmtv::pmt &pmtTimeUtcNs = parameters.at(std::string(gr::tag::TRIGGER_TIME.shortKey()));
            if (std::holds_alternative<uint64_t>(pmtTimeUtcNs)) {
                ctx.time = std::get<uint64_t>(pmtTimeUtcNs);
            }
        }

        if (ctx.time == 0ULL) {
            ctx.time = settings::convertTimePointToUint64Ns(std::chrono::system_clock::now());
        }
        return ctx;
    }

    void
    storeDefaultSettings(property_map &oldSettings) {
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
    inline constexpr static void
    processMembers(Func func) {
        if constexpr (detail::HasBaseType<T>) {
            refl::util::for_each(refl::reflect<typename std::remove_cvref_t<T>::base_t>().members, func);
        }
        refl::util::for_each(refl::reflect<T>().members, func);
    }
};

static_assert(SettingsLike<CtxSettings<int>>);

} // namespace gr

#endif // GNURADIO_TRANSACTIONS_HPP
