#ifndef GRAPH_PROTOTYPE_SETTINGS_HPP
#define GRAPH_PROTOTYPE_SETTINGS_HPP

#include "annotated.hpp"
#include <atomic>
#include <chrono>
#include <concepts>
#include <mutex>
#include <optional>
#include <reflection.hpp>
#include <set>
#include <tag.hpp>
#include <variant>

namespace fair::graph {

struct SettingsCtx {
    // using TimePoint = std::chrono::time_point<std::chrono::utc_clock>; // TODO: change once the C++20 support is ubiquitous
    using TimePoint               = std::chrono::time_point<std::chrono::system_clock>;
    std::optional<TimePoint> time = std::nullopt; /// UTC time-stamp from which the setting is valid
    property_map             context;             /// user-defined multiplexing context for which the setting is valid
};

template<typename T>
concept Settings = requires(T t, std::span<const std::string> parameter_keys, const std::string &parameter_key, const property_map &parameters, SettingsCtx ctx) {
    /**
     * @brief returns if there are stages settings that haven't been applied yet.
     */
    { t.changed() } -> std::same_as<bool>;

    /**
     * @brief stages new key-value pairs that shall replace the block field-based settings.
     * N.B. settings become only active after executing 'apply_staged_parameters()' (usually done early on in the 'node::work()' function)
     * @return key-value pairs that could not be set
     */
    { t.set(parameters, ctx) } -> std::same_as<property_map>;
    { t.set(parameters) } -> std::same_as<property_map>;

    /**
     * @brief updates parameters based on node input tags for those with keys stored in `auto_update_parameters()`
     * Parameter changes to down-stream nodes is controlled via `auto_forward_parameters()`
     */
    { t.auto_update(parameters, ctx) } -> std::same_as<void>;
    { t.auto_update(parameters) } -> std::same_as<void>;

    /**
     * @brief return all available node settings as key-value pairs
     */
    { t.get() } -> std::same_as<property_map>;

    /**
     * @brief return key-pmt values map for multiple keys
     */
    { t.get(parameter_keys, ctx) } -> std::same_as<property_map>;
    { t.get(parameter_keys) } -> std::same_as<property_map>;

    /**
     * @brief return pmt value for a single key
     */
    { t.get(parameter_key, ctx) } -> std::same_as<std::optional<pmtv::pmt>>;
    { t.get(parameter_key) } -> std::same_as<std::optional<pmtv::pmt>>;

    /**
     * @brief returns the staged/not-yet-applied new parameters
     */
    { t.staged_parameters() } -> std::same_as<const property_map>;

    /**
     * @brief synchronise map-based with actual node field-based settings
     */
    { t.apply_staged_parameters() } -> std::same_as<const property_map>;

    /**
     * @brief synchronises the map-based with the node's field-based parameters
     * (N.B. usually called after the staged parameters have been synchronised)
     */
    { t.update_active_parameters() } -> std::same_as<void>;
};

struct settings_base {
    std::atomic_bool _changed{ false };

    virtual ~settings_base() = default;

    void
    swap(settings_base &other) noexcept {
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
    [[nodiscard]] bool
    changed() const noexcept {
        return _changed;
    }

    /**
     * @brief stages new key-value pairs that shall replace the block field-based settings.
     * N.B. settings become only active after executing 'apply_staged_parameters()' (usually done early on in the 'node::work()' function)
     * @return key-value pairs that could not be set
     */
    [[nodiscard]] virtual property_map
    set(const property_map &parameters, SettingsCtx ctx = {})
            = 0;

    /**
     * @brief updates parameters based on node input tags for those with keys stored in `auto_update_parameters()`
     * Parameter changes to down-stream nodes is controlled via `auto_forward_parameters()`
     */
    virtual void
    auto_update(const property_map &parameters, SettingsCtx = {})
            = 0;

    /**
     * @brief return all (or for selected multiple keys) available node settings as key-value pairs
     */
    [[nodiscard]] virtual property_map
    get(std::span<const std::string> parameter_keys = {}, SettingsCtx = {}) const noexcept
            = 0;

    [[nodiscard]] virtual std::optional<pmtv::pmt>
    get(const std::string &parameter_key, SettingsCtx = {}) const noexcept = 0;

    /**
     * @brief returns the staged/not-yet-applied new parameters
     */
    [[nodiscard]] virtual const property_map
    staged_parameters() const
            = 0;

    [[nodiscard]] virtual std::set<std::string, std::less<>> &
    auto_update_parameters() noexcept
            = 0;

    [[nodiscard]] virtual std::set<std::string, std::less<>> &
    auto_forward_parameters() noexcept
            = 0;

    /**
     * @brief synchronise map-based with actual node field-based settings
     * returns map with key-value tags that should be forwarded
     * to dependent/child nodes.
     */
    [[nodiscard]] virtual const property_map
    apply_staged_parameters() noexcept
            = 0;

    /**
     * @brief synchronises the map-based with the node's field-based parameters
     * (N.B. usually called after the staged parameters have been synchronised)
     */
    virtual void
    update_active_parameters() noexcept
            = 0;
};

namespace detail {
template<typename T>
concept HasBaseType = requires { typename std::remove_cvref_t<T>::base_t; };
};

template<typename Node>
class basic_settings : public settings_base {
    Node                              *_node = nullptr;
    mutable std::mutex                 _lock{};
    property_map                       _active{}; // copy of class field settings as pmt-style map
    property_map                       _staged{}; // parameters to become active before the next work() call
    std::set<std::string, std::less<>> _auto_update{};
    std::set<std::string, std::less<>> _auto_forward{};

public:
    basic_settings()  = delete;
    ~basic_settings() = default;

    explicit constexpr basic_settings(Node &node) noexcept : settings_base(), _node(&node) {
        if constexpr (refl::is_reflectable<Node>()) {
            meta::tuple_for_each(
                    [this](auto &&default_tag) {
                        auto iterate_over_member = [&](auto member) {
                            using RawType = std::remove_cvref_t<decltype(member(*_node))>;
                            using Type    = unwrap_if_wrapped_t<RawType>;
                            if constexpr (is_writable(member) && (std::is_arithmetic_v<Type> || std::is_same_v<Type, std::string> || fair::meta::vector_type<Type>) ) {
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
                        if constexpr (detail::HasBaseType<Node>) {
                            refl::util::for_each(refl::reflect<typename std::remove_cvref_t<Node>::base_t>().members, iterate_over_member);
                        }
                        refl::util::for_each(refl::reflect<Node>().members, iterate_over_member);
                    },
                    fair::graph::tag::DEFAULT_TAGS);

            // handle meta-information for UI and other non-processing-related purposes
            auto iterate_over_member = [&]<typename Member>(Member member) {
                using RawType = std::remove_cvref_t<decltype(member(*_node))>;
                // disable clang format because v16 cannot handle in-line requires clauses with return types nicely yet
                // clang-format off
                if constexpr (requires(Node t) { t.meta_information; }) {
                    static_assert(std::is_same_v<unwrap_if_wrapped_t<decltype(_node->meta_information)>, property_map>);
                    if constexpr (requires(Node t) { t.description; }) {
                        static_assert(std::is_same_v<std::remove_cvref_t<unwrap_if_wrapped_t<decltype(Node::description)>>, std::string_view>);
                        _node->meta_information.value["description"] = std::string(_node->description);
                    }

                    if constexpr (AnnotatedType<RawType>) {
                        _node->meta_information.value[fmt::format("{}::description", get_display_name(member))] = std::string(RawType::description());
                        _node->meta_information.value[fmt::format("{}::documentation", get_display_name(member))] = std::string(RawType::documentation());
                        _node->meta_information.value[fmt::format("{}::unit", get_display_name(member))] = std::string(RawType::unit());
                        _node->meta_information.value[fmt::format("{}::visible", get_display_name(member))] = RawType::visible();
                    }
                }
                // clang-format on
            };
            if constexpr (detail::HasBaseType<Node>) {
                refl::util::for_each(refl::reflect<typename std::remove_cvref_t<Node>::base_t>().members, iterate_over_member);
            }
            refl::util::for_each(refl::reflect<Node>().members, iterate_over_member);
        }
    }

    constexpr basic_settings(const basic_settings &other) noexcept : settings_base(other) {
        basic_settings temp(other);
        swap(temp);
    }

    constexpr basic_settings(basic_settings &&other) noexcept : settings_base(std::move(other)) {
        basic_settings temp(std::move(other));
        swap(temp);
    }

    basic_settings &
    operator=(const basic_settings &other) noexcept {
        swap(other);
        return *this;
    }

    basic_settings &
    operator=(basic_settings &&other) noexcept {
        basic_settings temp(std::move(other));
        swap(temp);
        return *this;
    }

    void
    swap(basic_settings &other) noexcept {
        if (this == &other) {
            return;
        }
        settings_base::swap(other);
        std::swap(_node, other._node);
        std::scoped_lock lock(_lock, other._lock);
        std::swap(_active, other._active);
        std::swap(_staged, other._staged);
        std::swap(_auto_update, other._auto_update);
        std::swap(_auto_forward, other._auto_forward);
    }

    [[nodiscard]] property_map
    set(const property_map &parameters, SettingsCtx = {}) override {
        property_map ret;
        if constexpr (refl::is_reflectable<Node>()) {
            std::lock_guard lg(_lock);
            for (const auto &[localKey, localValue] : parameters) {
                const auto &key                 = localKey;
                const auto &value               = localValue;
                bool        is_set              = false;
                auto        iterate_over_member = [&, this](auto member) {
                    using Type = unwrap_if_wrapped_t<std::remove_cvref_t<decltype(member(*_node))>>;
                    if constexpr (is_writable(member) && (std::is_arithmetic_v<Type> || std::is_same_v<Type, std::string> || fair::meta::vector_type<Type>) ) {
                        if (std::string(get_display_name(member)) == key && std::holds_alternative<Type>(value)) {
                            if (_auto_update.contains(key)) {
                                _auto_update.erase(key);
                            }
                            _staged.insert_or_assign(key, value);
                            settings_base::_changed.store(true);
                            is_set = true;
                        }
                    }
                };
                if constexpr (detail::HasBaseType<Node>) {
                    refl::util::for_each(refl::reflect<typename std::remove_cvref_t<Node>::base_t>().members, iterate_over_member);
                }
                refl::util::for_each(refl::reflect<Node>().members, iterate_over_member);
                if (!is_set) {
                    ret.insert_or_assign(key, pmtv::pmt(value));
                }
            }
        }

        // copy items that could not be matched to the node's meta_information map (if available)
        if constexpr (requires(Node t) {
                          {
                              unwrap_if_wrapped_t<decltype(t.meta_information)> {}
                          } -> std::same_as<property_map>;
                      }) {
            update_maps(ret, _node->meta_information);
        }

        return ret; // N.B. returns those <key:value> parameters that could not be set
    }

    void
    auto_update(const property_map &parameters, SettingsCtx = {}) override {
        if constexpr (refl::is_reflectable<Node>()) {
            for (const auto &[localKey, localValue] : parameters) {
                const auto &key                 = localKey;
                const auto &value               = localValue;
                auto        iterate_over_member = [&](auto member) {
                    using Type = unwrap_if_wrapped_t<std::remove_cvref_t<decltype(member(*_node))>>;
                    if constexpr (is_writable(member) && (std::is_arithmetic_v<Type> || std::is_same_v<Type, std::string> || fair::meta::vector_type<Type>) ) {
                        if (std::string(get_display_name(member)) == key && std::holds_alternative<Type>(value)) {
                            _staged.insert_or_assign(key, value);
                            settings_base::_changed.store(true);
                        }
                    }
                };
                if constexpr (detail::HasBaseType<Node>) {
                    refl::util::for_each(refl::reflect<typename std::remove_cvref_t<Node>::base_t>().members, iterate_over_member);
                }
                refl::util::for_each(refl::reflect<Node>().members, iterate_over_member);
            }
        }
    }

    [[nodiscard]] const property_map
    staged_parameters() const noexcept override {
        std::lock_guard lg(_lock);
        return _staged;
    }

    [[nodiscard]] property_map
    get(std::span<const std::string> parameter_keys = {}, SettingsCtx = {}) const noexcept override {
        std::lock_guard lg(_lock);
        property_map    ret;
        if (parameter_keys.empty()) {
            ret = _active;
            return ret;
        }
        for (const auto &key : parameter_keys) {
            if (_active.contains(key)) {
                ret.insert_or_assign(key, _active.at(key));
            }
        }
        return ret;
    }

    [[nodiscard]] std::optional<pmtv::pmt>
    get(const std::string &parameter_key, SettingsCtx = {}) const noexcept override {
        if constexpr (refl::is_reflectable<Node>()) {
            std::lock_guard lg(_lock);

            if (_active.contains(parameter_key)) {
                return { _active.at(parameter_key) };
            }
        }

        return std::nullopt;
    }

    [[nodiscard]] std::set<std::string, std::less<>> &
    auto_update_parameters() noexcept override {
        return _auto_update;
    }

    [[nodiscard]] std::set<std::string, std::less<>> &
    auto_forward_parameters() noexcept override {
        return _auto_forward;
    }

    /**
     * @brief synchronise map-based with actual node field-based settings
     * returns map with key-value tags that should be forwarded
     * to dependent/child nodes.
     */
    [[nodiscard]] const property_map
    apply_staged_parameters() noexcept override {
        property_map forward_parameters; // parameters that should be forwarded to dependent child nodes
        if constexpr (refl::is_reflectable<Node>()) {
            std::lock_guard lg(_lock);

            property_map    oldSettings;
            if constexpr (requires(Node d, const property_map &map) { d.settings_changed(map, map); }) {
                // take a copy of the field -> map value of the old settings
                if constexpr (refl::is_reflectable<Node>()) {
                    auto iterate_over_member = [&, this](auto member) {
                        using Type = unwrap_if_wrapped_t<std::remove_cvref_t<decltype(member(*_node))>>;

                        if constexpr (is_readable(member) && (std::integral<Type> || std::floating_point<Type> || std::is_same_v<Type, std::string> || fair::meta::vector_type<Type>) ) {
                            oldSettings.insert_or_assign(get_display_name(member), pmtv::pmt(member(*_node)));
                        }
                    };
                    if constexpr (detail::HasBaseType<Node>) {
                        refl::util::for_each(refl::reflect<typename std::remove_cvref_t<Node>::base_t>().members, iterate_over_member);
                    }
                    refl::util::for_each(refl::reflect<Node>().members, iterate_over_member);
                }
            }

            property_map staged;
            for (const auto &[localKey, localStaged_value] : _staged) {
                const auto &key                 = localKey;
                const auto &staged_value        = localStaged_value;
                auto        iterate_over_member = [&key, &staged, &forward_parameters, &staged_value, this](auto member) {
                    using Type = unwrap_if_wrapped_t<std::remove_cvref_t<decltype(member(*_node))>>;
                    if constexpr (is_writable(member) && (std::integral<Type> || std::floating_point<Type> || std::is_same_v<Type, std::string> || fair::meta::vector_type<Type>) ) {
                        if (std::string(get_display_name(member)) == key && std::holds_alternative<Type>(staged_value)) {
                            member(*_node) = std::get<Type>(staged_value);
                            if constexpr (requires { _node->settings_changed(/* old settings */ _active, /* new settings */ staged); }) {
                                staged.insert_or_assign(key, staged_value);
                            } else {
                                std::ignore = staged; // help clang to see why staged is not unused
                            }
                            if (_auto_forward.contains(get_display_name(member))) {
                                forward_parameters.insert_or_assign(key, staged_value);
                            }
                        }
                    }
                };
                if constexpr (detail::HasBaseType<Node>) {
                    refl::util::for_each(refl::reflect<typename std::remove_cvref_t<Node>::base_t>().members, iterate_over_member);
                }
                refl::util::for_each(refl::reflect<Node>().members, iterate_over_member);
            }
            auto iterate_over_member = [&, this](auto member) {
                using Type = unwrap_if_wrapped_t<std::remove_cvref_t<decltype(member(*_node))>>;

                if constexpr (is_readable(member) && (std::integral<Type> || std::floating_point<Type> || std::is_same_v<Type, std::string> || fair::meta::vector_type<Type>) ) {
                    _active.insert_or_assign(get_display_name(member), pmtv::pmt(member(*_node)));
                }
            };
            if constexpr (detail::HasBaseType<Node>) {
                refl::util::for_each(refl::reflect<typename std::remove_cvref_t<Node>::base_t>().members, iterate_over_member);
            }
            refl::util::for_each(refl::reflect<Node>().members, iterate_over_member);

            if constexpr (requires(Node d, const property_map &map) { d.settings_changed(map, map); }) {
                if (!staged.empty()) {
                    _node->settings_changed(/* old settings */ oldSettings, /* new settings */ staged);
                }
            }
            _staged.clear();
        }
        settings_base::_changed.store(false);
        return forward_parameters;
    }

    void
    update_active_parameters() noexcept override {
        if constexpr (refl::is_reflectable<Node>()) {
            std::lock_guard lg(_lock);
            auto            iterate_over_member = [&, this](auto member) {
                using Type = unwrap_if_wrapped_t<std::remove_cvref_t<decltype(member(*_node))>>;

                if constexpr (is_readable(member) && (std::integral<Type> || std::floating_point<Type> || std::is_same_v<Type, std::string>) ) {
                    _active.insert_or_assign(get_display_name_const(member).str(), member(*_node));
                }
            };
            if constexpr (detail::HasBaseType<Node>) {
                refl::util::for_each(refl::reflect<typename std::remove_cvref_t<Node>::base_t>().members, iterate_over_member);
            }
            refl::util::for_each(refl::reflect<Node>().members, iterate_over_member);
        }
    }
};

static_assert(Settings<basic_settings<int>>);

} // namespace fair::graph
#endif // GRAPH_PROTOTYPE_SETTINGS_HPP
