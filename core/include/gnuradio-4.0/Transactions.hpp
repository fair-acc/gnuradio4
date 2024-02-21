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
    using MatchPredicate = std::function<std::optional<bool>(const property_map &, const property_map &, std::size_t)>;

    TBlock                                       *_block = nullptr;
    mutable std::mutex                            _lock{};
    property_map                                  _active{};
    property_map                                  _staged{};
    std::set<std::string, std::less<>>            _auto_update{};
    std::set<std::string, std::less<>>            _auto_forward{};
    std::unordered_map<SettingsCtx, property_map> _settings{};
    property_map                                  _default_settings{};
    MatchPredicate                                _match_pred = nullMatchPred;

public:
    explicit CtxSettings(TBlock &block, MatchPredicate matchPred = nullMatchPred) noexcept : SettingsBase(), _block(&block), _match_pred(matchPred) {
        if constexpr (requires { &TBlock::settingsChanged; }) { // if settingsChanged is defined
            static_assert(HasSettingsChangedCallback<TBlock>, "if provided, settingsChanged must have either a `(const property_map& old, property_map& new, property_map& fwd)`"
                                                              "or `(const property_map& old, property_map& new)` paremeter signatures.");
        }

        if constexpr (requires { &TBlock::reset; }) { // if reset is defined
            static_assert(HasSettingsResetCallback<TBlock>, "if provided, reset() may have no function parameters");
        }

        if constexpr (refl::is_reflectable<TBlock>()) {
            meta::tuple_for_each(
                    [this](auto &&default_tag) {
                        auto iterate_over_member = [&](auto member) {
                            using RawType = std::remove_cvref_t<decltype(member(*_block))>;
                            using Type    = unwrap_if_wrapped_t<RawType>;
                            if constexpr (!std::is_const_v<Type> && is_writable(member) && (std::is_arithmetic_v<Type> || std::is_same_v<Type, std::string> || gr::meta::vector_type<Type>) ) {
                                auto matchesIgnoringPrefix = [](std::string_view str, std::string_view prefix, std::string_view target) {
                                    if (str.starts_with(prefix)) {
                                        str.remove_prefix(prefix.size());
                                    }
                                    return str == target;
                                };
                                if (matchesIgnoringPrefix(default_tag.shortKey(), std::string_view(GR_TAG_PREFIX), get_display_name(member))) {
                                    _auto_forward.emplace(get_display_name(member));
                                }
                                _auto_update.emplace(get_display_name(member));
                            }
                        };
                        if constexpr (detail::HasBaseType<TBlock>) {
                            refl::util::for_each(refl::reflect<typename std::remove_cvref_t<TBlock>::base_t>().members, iterate_over_member);
                        }
                        refl::util::for_each(refl::reflect<TBlock>().members, iterate_over_member);
                    },
                    gr::tag::DEFAULT_TAGS);

            // handle meta-information for UI and other non-processing-related purposes
            auto iterate_over_member = [&]<typename Member>(Member member) {
                using RawType = std::remove_cvref_t<decltype(member(*_block))>;
                // disable clang format because v16 cannot handle in-line requires clauses with return types nicely yet
                // clang-format off
                if constexpr (requires(TBlock t) { t.metaInformation; }) {
                    static_assert(std::is_same_v<unwrap_if_wrapped_t<decltype(_block->metaInformation)>, property_map>);
                    if constexpr (requires(TBlock t) { t.description; }) {
                        static_assert(std::is_same_v<std::remove_cvref_t<unwrap_if_wrapped_t<decltype(TBlock::description)>>, std::string_view>);
                        _block->metaInformation.value["description"] = std::string(_block->description);
                    }

                    if constexpr (AnnotatedType<RawType>) {
                        _block->metaInformation.value[fmt::format("{}::description", get_display_name(member))] = std::string(RawType::description());
                        _block->metaInformation.value[fmt::format("{}::documentation", get_display_name(member))] = std::string(RawType::documentation());
                        _block->metaInformation.value[fmt::format("{}::unit", get_display_name(member))] = std::string(RawType::unit());
                        _block->metaInformation.value[fmt::format("{}::visible", get_display_name(member))] = RawType::visible();
                    }
                }
                // clang-format on
            };
            if constexpr (detail::HasBaseType<TBlock>) {
                refl::util::for_each(refl::reflect<typename std::remove_cvref_t<TBlock>::base_t>().members, iterate_over_member);
            }
            refl::util::for_each(refl::reflect<TBlock>().members, iterate_over_member);
        }
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
        std::swap(_active, other._active);
        std::swap(_staged, other._staged);
        std::swap(_settings, other._settings);
        std::swap(_auto_update, other._auto_update);
        std::swap(_auto_forward, other._auto_forward);
        std::swap(_match_pred, other._match_pred);
    }

    [[nodiscard]] property_map
    set(const property_map &parameters, SettingsCtx ctx = {}) override {
        property_map ret;
        if constexpr (refl::is_reflectable<TBlock>()) {
            std::lock_guard lg(_lock);
            for (const auto &[localKey, localValue] : parameters) {
                const auto &key                 = localKey;
                const auto &value               = localValue;
                bool        is_set              = false;
                auto        iterate_over_member = [&, this](auto member) {
                    using Type = unwrap_if_wrapped_t<std::remove_cvref_t<decltype(member(*_block))>>;
                    if constexpr (!std::is_const_v<Type> && is_writable(member) && (std::is_arithmetic_v<Type> || std::is_same_v<Type, std::string> || gr::meta::vector_type<Type>) ) {
                        if (std::string(get_display_name(member)) == key && std::holds_alternative<Type>(value)) {
                            if (_auto_update.contains(key)) {
                                _auto_update.erase(key);
                            }
                            SettingsBase::_changed.store(true);
                            is_set = true;
                        }
                    }
                };
                if constexpr (detail::HasBaseType<TBlock>) {
                    refl::util::for_each(refl::reflect<typename std::remove_cvref_t<TBlock>::base_t>().members, iterate_over_member);
                }
                refl::util::for_each(refl::reflect<TBlock>().members, iterate_over_member);
                if (!is_set) {
                    ret.insert_or_assign(key, pmtv::pmt(value));
                }
            }
        }

        // copy items that could not be matched to the node's meta_information map (if available)
        if constexpr (requires(TBlock t) {
                          {
                              unwrap_if_wrapped_t<decltype(t.metaInformation)> {}
                          } -> std::same_as<property_map>;
                      }) {
            updateMaps(ret, _block->metaInformation);
        }

        _settings[ctx] = parameters;
        _settings[{}]  = parameters;

        return ret; // N.B. returns those <key:value> parameters that could not be set
    }

    void
    storeDefaults() override {
        this->storeDefaultSettings(_default_settings);
    }

    void
    resetDefaults() override {
        _settings.clear();
        _staged     = _default_settings;
        std::ignore = applyStagedParameters();
        if constexpr (HasSettingsResetCallback<TBlock>) {
            _block->reset();
        }
    }

    void
    autoUpdate(const property_map &parameters, SettingsCtx = {}) override {
        if constexpr (refl::is_reflectable<TBlock>()) {
            for (const auto &[localKey, localValue] : parameters) {
                const auto &key                 = localKey;
                const auto &value               = localValue;
                auto        iterate_over_member = [&](auto member) {
                    using Type = unwrap_if_wrapped_t<std::remove_cvref_t<decltype(member(*_block))>>;
                    if constexpr (!std::is_const_v<Type> && is_writable(member) && (std::is_arithmetic_v<Type> || std::is_same_v<Type, std::string> || gr::meta::vector_type<Type>) ) {
                        if (std::string(get_display_name(member)) == key && std::holds_alternative<Type>(value)) {
                            _staged.insert_or_assign(key, value);
                            SettingsBase::_changed.store(true);
                        }
                    }
                };
                if constexpr (detail::HasBaseType<TBlock>) {
                    refl::util::for_each(refl::reflect<typename std::remove_cvref_t<TBlock>::base_t>().members, iterate_over_member);
                }
                refl::util::for_each(refl::reflect<TBlock>().members, iterate_over_member);
            }
        }
    }

    [[nodiscard]] const property_map
    stagedParameters() const noexcept override {
        std::lock_guard lg(_lock);
        return _staged;
    }

    [[nodiscard]] property_map
    get(std::span<const std::string> parameter_keys = {}, SettingsCtx ctx = {}) const noexcept override {
        std::lock_guard lg(_lock);
        property_map    ret;

        if (_settings.empty()) {
            return ret;
        }

        if (_settings.contains(ctx)) {
            // is there an exact match?
            const auto &exact_match = _settings.at(ctx);
            ret                     = exact_match;
        } else {
            // try the match predicate instead
            const auto &match = bestMatch(ctx.context);
            ret               = match.value_or(ret);
        }

        // return only the needed values
        std::ignore = std::erase_if(ret, [&](const auto &i) { return std::find(parameter_keys.begin(), parameter_keys.end(), i.first) == parameter_keys.end(); });
        return ret;
    }

    [[nodiscard]] std::optional<pmtv::pmt>
    get(const std::string &parameter_key, SettingsCtx ctx = {}) const noexcept override {
        auto res = get(std::array<std::string, 1>({ parameter_key }), ctx);
        if (res.contains(parameter_key)) {
            return res.at(parameter_key);
        } else {
            return std::nullopt;
        }
    }

    [[nodiscard]] std::set<std::string, std::less<>> &
    autoUpdateParameters() noexcept override {
        return _auto_update;
    }

    [[nodiscard]] std::set<std::string, std::less<>> &
    autoForwardParameters() noexcept override {
        return _auto_forward;
    }

    [[nodiscard]] ApplyStagedParametersResult
    applyStagedParameters() noexcept override {
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
            for (const auto &[localKey, localStaged_value] : _staged) {
                const auto &key                  = localKey;
                const auto &staged_value         = localStaged_value;
                auto        apply_member_changes = [&key, &staged, &result, &staged_value, this](auto member) {
                    using RawType = std::remove_cvref_t<decltype(member(*_block))>;
                    using Type    = unwrap_if_wrapped_t<RawType>;
                    if constexpr (traits::port::is_not_any_port_or_collection<Type> && !std::is_const_v<Type> && is_writable(member) && settings::isSupportedType<Type>()) {
                        if (std::string(get_display_name(member)) == key && std::holds_alternative<Type>(staged_value)) {
                            if constexpr (is_annotated<RawType>()) {
                                if (member(*_block).validate_and_set(std::get<Type>(staged_value))) {
                                    if constexpr (HasSettingsChangedCallback<TBlock>) {
                                        staged.insert_or_assign(key, staged_value);
                                    } else {
                                        std::ignore = staged; // help clang to see why staged is not unused
                                    }
                                } else {
                                    // TODO: replace with pmt error message on msgOut port (to note: clang compiler bug/issue)
#if !defined(__EMSCRIPTEN__) && !defined(__clang__)
                                    fmt::print(stderr, " cannot set field {}({})::{} = {} to {} due to limit constraints [{}, {}] validate func is {} defined\n", //
                                               _block->unique_name, _block->name, member(*_block), std::get<Type>(staged_value),                                  //
                                               std::string(get_display_name(member)), RawType::LimitType::MinRange,
                                               RawType::LimitType::MaxRange, //
                                               RawType::LimitType::ValidatorFunc == nullptr ? "not" : "");
#else
                                    fmt::print(stderr, " cannot set field {}({})::{} = {} to {} due to limit constraints [{}, {}] validate func is {} defined\n", //
                                               "_block->uniqueName", "_block->name", member(*_block), std::get<Type>(staged_value),                               //
                                               std::string(get_display_name(member)), RawType::LimitType::MinRange,
                                               RawType::LimitType::MaxRange, //
                                               RawType::LimitType::ValidatorFunc == nullptr ? "not" : "");
#endif
                                }
                            } else {
                                member(*_block) = std::get<Type>(staged_value);
                                result.appliedParameters.insert_or_assign(key, staged_value);
                                if constexpr (HasSettingsChangedCallback<TBlock>) {
                                    staged.insert_or_assign(key, staged_value);
                                } else {
                                    std::ignore = staged; // help clang to see why staged is not unused
                                }
                            }
                        }
                        if (_auto_forward.contains(key)) {
                            result.forwardParameters.insert_or_assign(key, staged_value);
                        }
                    }
                };
                processMembers<TBlock>(apply_member_changes);
            }

            // update active parameters
            auto update_active = [this](auto member) {
                using Type = unwrap_if_wrapped_t<std::remove_cvref_t<decltype(member(*_block))>>;
                if constexpr (is_readable(member) && settings::isSupportedType<Type>()) {
                    _active.insert_or_assign(get_display_name(member), pmtv::pmt(member(*_block)));
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

    void
    updateActiveParameters() noexcept override {
        if constexpr (refl::is_reflectable<TBlock>()) {
            std::lock_guard lg(_lock);
            auto            iterate_over_member = [&, this](auto member) {
                using Type = unwrap_if_wrapped_t<std::remove_cvref_t<decltype(member(*_block))>>;

                if constexpr (is_readable(member) && settings::isSupportedType<Type>()) {
                    _active.insert_or_assign(get_display_name_const(member).str(), member(*_block));
                }
            };
            if constexpr (detail::HasBaseType<TBlock>) {
                refl::util::for_each(refl::reflect<typename std::remove_cvref_t<TBlock>::base_t>().members, iterate_over_member);
            }
            refl::util::for_each(refl::reflect<TBlock>().members, iterate_over_member);
        }
    }

private:
    std::optional<property_map>
    bestMatch(const property_map &context) const {
        // retry until we either get a match or std::nullopt
        for (std::size_t attempt = 0;; ++attempt) {
            for (const auto &i : _settings) {
                const auto matchres = _match_pred(i.first.context, context, attempt);
                if (!matchres) {
                    return std::nullopt;
                } else if (*matchres) {
                    return i.second;
                }
            }
        }
    }

    void
    storeDefaultSettings(property_map &oldSettings) {
        // take a copy of the field -> map value of the old settings
        if constexpr (refl::is_reflectable<TBlock>()) {
            auto iterate_over_member = [&, this](auto member) {
                using Type = unwrap_if_wrapped_t<std::remove_cvref_t<decltype(member(*_block))>>;

                if constexpr (is_readable(member) && settings::isSupportedType<Type>()) {
                    oldSettings.insert_or_assign(get_display_name(member), pmtv::pmt(member(*_block)));
                }
            };
            if constexpr (detail::HasBaseType<TBlock>) {
                refl::util::for_each(refl::reflect<typename std::remove_cvref_t<TBlock>::base_t>().members, iterate_over_member);
            }
            refl::util::for_each(refl::reflect<TBlock>().members, iterate_over_member);
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
