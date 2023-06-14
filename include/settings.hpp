#ifndef GRAPH_PROTOTYPE_SETTINGS_HPP
#define GRAPH_PROTOTYPE_SETTINGS_HPP

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
    tag_t::map_type          context;             /// user-defined multiplexing context for which the setting is valid
};

template<typename T>
concept Settings = requires(T t, std::span<const std::string> parameter_keys, const std::string &parameter_key, const tag_t::map_type &parameters, SettingsCtx ctx) {
    /**
     * @brief returns if there are stages settings that haven't been applied yet.
     */
    { t.changed() } -> std::same_as<bool>;

    /**
     * @brief stages new key-value pairs that shall replace the block field-based settings.
     * N.B. settings become only active after executing 'apply_staged_parameters()' (usually done early on in the 'node::work()' function)
     * @return key-value pairs that could not be set
     */
    { t.set(parameters, ctx) } -> std::same_as<tag_t::map_type>;
    { t.set(parameters) } -> std::same_as<tag_t::map_type>;

    /**
     * @brief updates parameters based on node input tags for those with keys stored in `auto_update_parameters()`
     * Parameter changes to down-stream nodes is controlled via `auto_forward_parameters()`
     */
    { t.auto_update(parameters, ctx) } -> std::same_as<void>;
    { t.auto_update(parameters) } -> std::same_as<void>;

    /**
     * @brief return all available node settings as key-value pairs
     */
    { t.get() } -> std::same_as<tag_t::map_type>;

    /**
     * @brief return key-pmt values map for multiple keys
     */
    { t.get(parameter_keys, ctx) } -> std::same_as<tag_t::map_type>;
    { t.get(parameter_keys) } -> std::same_as<tag_t::map_type>;

    /**
     * @brief return pmt value for a single key
     */
    { t.get(parameter_key, ctx) } -> std::same_as<std::optional<pmtv::pmt>>;
    { t.get(parameter_key) } -> std::same_as<std::optional<pmtv::pmt>>;

    /**
     * @brief returns the staged/not-yet-applied new parameters
     */
    { t.staged_parameters() } -> std::same_as<const tag_t::map_type>;

    /**
     * @brief synchronise map-based with actual node field-based settings
     */
    { t.apply_staged_parameters() } -> std::same_as<const tag_t::map_type>;

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
    [[nodiscard]] virtual tag_t::map_type
    set(const tag_t::map_type &parameters, SettingsCtx ctx = {})
            = 0;

    /**
     * @brief updates parameters based on node input tags for those with keys stored in `auto_update_parameters()`
     * Parameter changes to down-stream nodes is controlled via `auto_forward_parameters()`
     */
    virtual void
    auto_update(const tag_t::map_type &parameters, SettingsCtx = {})
            = 0;

    /**
     * @brief return all (or for selected multiple keys) available node settings as key-value pairs
     */
    [[nodiscard]] virtual tag_t::map_type
    get(std::span<const std::string> parameter_keys = {}, SettingsCtx = {}) const noexcept
            = 0;

    [[nodiscard]] virtual std::optional<pmtv::pmt>
    get(const std::string &parameter_key, SettingsCtx = {}) const noexcept = 0;

    /**
     * @brief returns the staged/not-yet-applied new parameters
     */
    [[nodiscard]] virtual const tag_t::map_type
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
    [[nodiscard]] virtual const tag_t::map_type
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

template<typename Node>
class basic_settings : public settings_base {
    Node                              *_node = nullptr;
    mutable std::mutex                 _lock{};
    tag_t::map_type                    _active{}; // copy of class field settings as pmt-style map
    tag_t::map_type                    _staged{}; // parameters to become active before the next work() call
    std::set<std::string, std::less<>> _auto_update{};
    std::set<std::string, std::less<>> _auto_forward{};

public:
    basic_settings()  = delete;
    ~basic_settings() = default;

    explicit constexpr basic_settings(Node &node) noexcept : settings_base(), _node(&node) {
        if constexpr (refl::is_reflectable<Node>()) {
            meta::tuple_for_each(
                    [this](auto &&default_tag) {
                        for_each(refl::reflect(*_node).members, [&](auto member) {
                            using Type = unwrap_if_wrapped_t<std::remove_cvref_t<decltype(member(*_node))>>;
                            if constexpr (is_writable(member) && (std::is_arithmetic_v<Type> || std::is_same_v<Type, std::string> || fair::meta::vector_type<Type>) ) {
                                if (default_tag.shortKey().ends_with(get_display_name(member))) {
                                    _auto_forward.emplace(get_display_name(member));
                                }
                                _auto_update.emplace(get_display_name(member));
                            }
                        });
                    },
                    fair::graph::tag::DEFAULT_TAGS);
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

    [[nodiscard]] tag_t::map_type
    set(const tag_t::map_type &parameters, SettingsCtx = {}) override {
        tag_t::map_type ret;
        if constexpr (refl::is_reflectable<Node>()) {
            std::lock_guard lg(_lock);
            for (const auto &[localKey, localValue] : parameters) {
                const auto &key    = localKey;
                const auto &value  = localValue;
                bool        is_set = false;
                for_each(refl::reflect(*_node).members, [&, this](auto member) {
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
                });
                if (!is_set) {
                    ret.insert_or_assign(key, pmtv::pmt(value));
                }
            }
        }

        return ret; // N.B. returns those <key:value> parameters that could not be set
    }

    void
    auto_update(const tag_t::map_type &parameters, SettingsCtx = {}) override {
        if constexpr (refl::is_reflectable<Node>()) {
            for (const auto &[localKey, localValue] : parameters) {
                const auto &key   = localKey;
                const auto &value = localValue;
                for_each(refl::reflect(*_node).members, [&](auto member) {
                    using Type = unwrap_if_wrapped_t<std::remove_cvref_t<decltype(member(*_node))>>;
                    if constexpr (is_writable(member) && (std::is_arithmetic_v<Type> || std::is_same_v<Type, std::string> || fair::meta::vector_type<Type>) ) {
                        if (std::string(get_display_name(member)) == key && std::holds_alternative<Type>(value)) {
                            _staged.insert_or_assign(key, value);
                            settings_base::_changed.store(true);
                        }
                    }
                });
            }
        }
    }

    [[nodiscard]] const tag_t::map_type
    staged_parameters() const noexcept override {
        std::lock_guard lg(_lock);
        return _staged;
    }

    [[nodiscard]] tag_t::map_type
    get(std::span<const std::string> parameter_keys = {}, SettingsCtx = {}) const noexcept override {
        std::lock_guard lg(_lock);
        tag_t::map_type ret;
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
    [[nodiscard]] const tag_t::map_type
    apply_staged_parameters() noexcept override {
        tag_t::map_type forward_parameters; // parameters that should be forwarded to dependent child nodes
        if constexpr (refl::is_reflectable<Node>()) {
            std::lock_guard lg(_lock);

            tag_t::map_type oldSettings;
            if constexpr (requires(Node d, const tag_t::map_type &map) { d.init(map, map); }) {
                // take a copy of the field -> map value of the old settings
                if constexpr (refl::is_reflectable<Node>()) {
                    for_each(refl::reflect(*_node).members, [&, this](auto member) {
                        using Type = unwrap_if_wrapped_t<std::remove_cvref_t<decltype(member(*_node))>>;

                        if constexpr (is_readable(member) && (std::integral<Type> || std::floating_point<Type> || std::is_same_v<Type, std::string> || fair::meta::vector_type<Type>) ) {
                            oldSettings.insert_or_assign(get_display_name(member), pmtv::pmt(member(*_node)));
                        }
                    });
                }
            }

            tag_t::map_type staged;
            for (const auto &[localKey, localStaged_value] : _staged) {
                const auto &key          = localKey;
                const auto &staged_value = localStaged_value;
                for_each(refl::reflect(*_node).members, [&key, &staged, &forward_parameters, &staged_value, this](auto member) {
                    using Type = unwrap_if_wrapped_t<std::remove_cvref_t<decltype(member(*_node))>>;
                    if constexpr (is_writable(member) && (std::integral<Type> || std::floating_point<Type> || std::is_same_v<Type, std::string> || fair::meta::vector_type<Type>) ) {
                        if (std::string(get_display_name(member)) == key && std::holds_alternative<Type>(staged_value)) {
                            member(*_node) = std::get<Type>(staged_value);
                            if constexpr (requires { _node->init(/* old settings */ _active, /* new settings */ staged); }) {
                                staged.insert_or_assign(key, staged_value);
                            }
                            if (_auto_forward.contains(get_display_name(member))) {
                                forward_parameters.insert_or_assign(key, staged_value);
                            }
                        }
                    }
                });
            }
            for_each(refl::reflect(*_node).members, [&, this](auto member) {
                using Type = unwrap_if_wrapped_t<std::remove_cvref_t<decltype(member(*_node))>>;

                if constexpr (is_readable(member) && (std::integral<Type> || std::floating_point<Type> || std::is_same_v<Type, std::string> || fair::meta::vector_type<Type>) ) {
                    _active.insert_or_assign(get_display_name(member), pmtv::pmt(member(*_node)));
                }
            });
            if constexpr (requires(Node d, const tag_t::map_type &map) { d.init(map, map); }) {
                if (!staged.empty()) {
                    _node->init(/* old settings */ oldSettings, /* new settings */ staged);
                }
            }
            _staged.clear();
        }
        return forward_parameters;
    }

    void
    update_active_parameters() noexcept override {
        if constexpr (refl::is_reflectable<Node>()) {
            std::lock_guard lg(_lock);
            for_each(refl::reflect(*_node).members, [&, this](auto member) {
                using Type = unwrap_if_wrapped_t<std::remove_cvref_t<decltype(member(*_node))>>;

                if constexpr (is_readable(member) && (std::integral<Type> || std::floating_point<Type> || std::is_same_v<Type, std::string>) ) {
                    _active.insert_or_assign(get_display_name_const(member).str(), member(*_node));
                }
            });
        }
    }
};

static_assert(Settings<basic_settings<int>>);

} // namespace fair::graph
#endif // GRAPH_PROTOTYPE_SETTINGS_HPP
