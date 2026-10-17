#include <gnuradio-4.0/Settings.hpp>

namespace gr {

// --- Simple accessors ---

bool CtxSettingsBase::changed() const noexcept { return gr::atomic_ref(_changed).load_acquire(); }

void CtxSettingsBase::setChanged(bool b) noexcept { gr::atomic_ref(_changed).store_release(b); }

void CtxSettingsBase::setInitBlockParameters(const property_map& parameters) { _initBlockParameters = parameters; }

const SettingsCtx& CtxSettingsBase::activeContext() const noexcept { return _activeCtx; }

const property_map& CtxSettingsBase::defaultParameters() const noexcept { return _defaultParameters; }

const property_map& CtxSettingsBase::activeParameters() const noexcept { return _activeParameters; }

// --- get() overloads ---

property_map CtxSettingsBase::get(std::span<const std::string> parameterKeys) const noexcept {
    std::lock_guard lg(_mutex);
    if (parameterKeys.empty()) {
        return property_map{_activeParameters, _activeParameters.resource()};
    }
    return _activeParameters.project(parameterKeys | std::views::transform([](const std::string& k) { return std::string(gr::tag::settingsKey(convert_string_domain(k))); }));
}

std::optional<Value> CtxSettingsBase::get(const std::string& parameterKey) const noexcept {
    std::lock_guard lg(_mutex);
    if (auto v = _activeParameters.find_value(std::string(gr::tag::settingsKey(convert_string_domain(parameterKey))))) {
        return Value{*v, std::pmr::get_default_resource()};
    }
    return std::nullopt;
}

// --- getStored() overloads ---

std::optional<property_map> CtxSettingsBase::getStored(std::span<const std::string> parameterKeys, SettingsCtx ctx) const noexcept {
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
    return allBestMatchParameters->project(parameterKeys | std::views::transform([](const std::string& k) { return std::string(gr::tag::settingsKey(convert_string_domain(k))); }));
}

std::optional<Value> CtxSettingsBase::getStored(const std::string& parameterKey, SettingsCtx ctx) const noexcept {
    auto res = getStored(std::array<std::string, 1>({parameterKey}), ctx);
    if (!res) {
        return std::nullopt;
    }
    if (auto v = res->find_value(std::string(gr::tag::settingsKey(convert_string_domain(parameterKey))))) {
        return Value{*v, std::pmr::get_default_resource()};
    }
    return std::nullopt;
}

// --- Remaining getters ---

gr::Size_t CtxSettingsBase::getNStoredParameters() const noexcept {
    std::lock_guard lg(_mutex);
    gr::Size_t      nParameters{0};
    for (const auto& stored : _storedParameters) {
        nParameters += static_cast<gr::Size_t>(stored.second.size());
    }
    return nParameters;
}

gr::Size_t CtxSettingsBase::getNAutoUpdateParameters() const noexcept {
    std::lock_guard lg(_mutex);
    return static_cast<gr::Size_t>(_autoUpdateParameters.size());
}

CtxSettingsBase::StoredParametersMap CtxSettingsBase::getStoredAll() const noexcept { return _storedParameters; }

const property_map& CtxSettingsBase::stagedParameters() const {
    std::lock_guard lg(_mutex);
    return _stagedParameters;
}

std::pmr::vector<std::string_view> CtxSettingsBase::autoUpdateParameters(SettingsCtx ctx) noexcept {
    std::lock_guard lg(_mutex);
    const auto      bestMatchSettingsCtx = findBestMatchSettingsCtx(ctx);
    const auto      it                   = bestMatchSettingsCtx ? _autoUpdateParameters.find(*bestMatchSettingsCtx) : _autoUpdateParameters.end();
    if (it == _autoUpdateParameters.end()) {
        return {};
    }
    return std::views::zip(doGetAllWritableMembers(), it->second) | std::views::filter([](const auto& nameAndBit) { return static_cast<bool>(std::get<1>(nameAndBit)); }) | std::views::keys | std::ranges::to<std::pmr::vector<std::string_view>>();
}

// --- setStaged() ---

property_map CtxSettingsBase::setStaged(const property_map& parameters) {
    std::lock_guard lg(_mutex);
    return doSetStagedImpl(parameters);
}

// --- Context management ---

std::optional<SettingsCtx> CtxSettingsBase::activateContext(SettingsCtx ctx) {
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
            if (!_autoUpdateParameters.contains(bestMatchSettingsCtx.value())) {
                auto inherited                                      = getBestMatchAutoUpdateParameters(bestMatchSettingsCtx.value());
                _autoUpdateParameters[bestMatchSettingsCtx.value()] = inherited ? std::move(*inherited) : allWritableBits();
            }
            const AutoUpdateBits& currentAutoUpdateParams = _autoUpdateParameters.at(bestMatchSettingsCtx.value());

            const std::span<const std::string_view> sortedNames = doGetAllWritableMembers();
            const auto                              followsTags = [&](std::string_view name) {
                const auto it  = std::ranges::lower_bound(sortedNames, name);
                const auto row = static_cast<std::size_t>(it - sortedNames.begin());
                return it != sortedNames.end() && *it == name && row < currentAutoUpdateParams.size() && currentAutoUpdateParams[row];
            };
            property_map notAutoUpdateParams;
            for (const auto& pair : parameters.value()) {
                if (!followsTags(std::string_view{pair.first})) {
                    notAutoUpdateParams.insert(pair);
                }
            }

            std::ignore = doSetStagedImpl(std::move(notAutoUpdateParams));
            _activeCtx  = bestMatchSettingsCtx.value();
            setChanged(true);
        }
    } else {
        std::optional<property_map> _parameters = getBestMatchStoredParameters(ctx);
        if (_parameters) {
            auto& parameters = *_parameters;
            _stagedParameters.insert(parameters.begin(), parameters.end());
            _activeCtx = bestMatchSettingsCtx.value();
            setChanged(true);
        } else {
            return std::nullopt;
        }
    }

    return bestMatchSettingsCtx;
}

bool CtxSettingsBase::removeContext(SettingsCtx ctx) {
    if (ctx.context.empty()) {
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
        _storedParameters.erase(it);
    }

    if (_activeCtx.context == ctx.context) {
        std::ignore = activateContext(); // Activate default context
    }

    return true;
}

void CtxSettingsBase::deferNotification(const property_map& appliedParameters) {
    std::lock_guard lg(_mutex);
    for (const auto& [key, value] : appliedParameters) {
        _deferredNotification.insert_or_assign(key, value);
    }
    gr::atomic_ref(_notificationPending).store_release(true);
}

std::optional<property_map> CtxSettingsBase::takeDeferredNotification() {
    if (!gr::atomic_ref(_notificationPending).load_acquire()) {
        return std::nullopt;
    }
    std::lock_guard lg(_mutex);
    gr::atomic_ref(_notificationPending).store_release(false);
    property_map applied = _deferredNotification;
    _deferredNotification.clear();
    return applied;
}

// --- assignFrom ---

void CtxSettingsBase::assignFrom(const CtxSettingsBase& other) {
    std::scoped_lock lock(_mutex, other._mutex);
    gr::atomic_ref(_changed).store_release(gr::atomic_ref(other._changed).load_acquire());
    _storedParameters     = other._storedParameters;
    _defaultParameters    = other._defaultParameters;
    _initBlockParameters  = other._initBlockParameters;
    _autoUpdateParameters = other._autoUpdateParameters;
    _matchPred            = other._matchPred;
    _activeCtx            = other._activeCtx;
    _stagedParameters     = other._stagedParameters;
    _activeParameters     = other._activeParameters;
}

void CtxSettingsBase::assignFrom(CtxSettingsBase&& other) noexcept {
    std::scoped_lock lock(_mutex, other._mutex);
    gr::atomic_ref(_changed).store_release(gr::atomic_ref(other._changed).load_acquire());
    _storedParameters     = std::move(other._storedParameters);
    _defaultParameters    = std::move(other._defaultParameters);
    _initBlockParameters  = std::move(other._initBlockParameters);
    _autoUpdateParameters = std::move(other._autoUpdateParameters);
    _matchPred            = std::exchange(other._matchPred, settings::nullMatchPred);
    _activeCtx            = std::exchange(other._activeCtx, {});
    _stagedParameters     = std::move(other._stagedParameters);
    _activeParameters     = std::move(other._activeParameters);
}

// --- Private helpers: match/search ---

std::optional<std::string> CtxSettingsBase::findBestMatchCtx(std::string_view contextToSearch) const {
    if (_storedParameters.empty()) {
        return std::nullopt;
    }

    // exact match
    if (_storedParameters.find(contextToSearch) != _storedParameters.end()) {
        return std::string(contextToSearch);
    }

    // retry until we either get a match or std::nullopt
    for (std::size_t attempt = 0;; ++attempt) {
        for (const auto& i : _storedParameters) {
            const auto matches = _matchPred(i.first, contextToSearch, attempt);
            if (!matches) {
                return std::nullopt;
            } else if (*matches) {
                return std::string(std::string_view{i.first}); // return the best matched SettingsCtx.context
            }
        }
    }
    return std::nullopt;
}

std::optional<SettingsCtx> CtxSettingsBase::findBestMatchSettingsCtx(const SettingsCtx& ctx) const {
    const auto bestMatchCtx = findBestMatchCtx(ctx.context);
    if (bestMatchCtx == std::nullopt) {
        return std::nullopt;
    }
    auto storedIt = _storedParameters.find(std::string_view{bestMatchCtx.value()});
    if (storedIt == _storedParameters.end() || storedIt->second.empty()) {
        return std::nullopt;
    }
    const auto& vec = storedIt->second;
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

std::optional<property_map> CtxSettingsBase::getBestMatchStoredParameters(const SettingsCtx& ctx) const {
    const auto bestMatchSettingsCtx = findBestMatchSettingsCtx(ctx);
    if (bestMatchSettingsCtx == std::nullopt) {
        return std::nullopt;
    }
    auto vecIt = _storedParameters.find(std::string_view{bestMatchSettingsCtx.value().context});
    if (vecIt == _storedParameters.end()) {
        return std::nullopt;
    }
    const auto& vec        = vecIt->second;
    const auto  parameters = std::ranges::find_if(vec, [&](const CtxSettingsPair& contextSettings) { return contextSettings.context == bestMatchSettingsCtx.value(); });

    return parameters != vec.end() ? std::optional(parameters->settings) : std::nullopt;
}

AutoUpdateBits CtxSettingsBase::allWritableBits() const { return AutoUpdateBits(doGetAllWritableMembers().size(), true, _autoUpdateParameters.get_allocator().resource()); }

std::optional<AutoUpdateBits> CtxSettingsBase::getBestMatchAutoUpdateParameters(const SettingsCtx& ctx) const {
    const auto bestMatchSettingsCtx = findBestMatchSettingsCtx(ctx);
    if (bestMatchSettingsCtx == std::nullopt || !_autoUpdateParameters.contains(bestMatchSettingsCtx.value())) {
        return std::nullopt;
    } else {
        return _autoUpdateParameters.at(bestMatchSettingsCtx.value());
    }
}

// --- Private helpers: storage/expiry ---

void CtxSettingsBase::resolveDuplicateTimestamp(SettingsCtx& ctx) {
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

void CtxSettingsBase::addStoredParameters(const property_map& newParameters, const SettingsCtx& ctx) {
    if (!_autoUpdateParameters.contains(ctx)) {
        auto inherited             = getBestMatchAutoUpdateParameters(ctx);
        _autoUpdateParameters[ctx] = inherited ? std::move(*inherited) : allWritableBits();
    }

    auto                          ctxKey                 = std::pmr::string(ctx.context, _storedParameters.get_allocator().resource());
    std::vector<CtxSettingsPair>& sortedVectorForContext = _storedParameters[std::move(ctxKey)];
    // binary search and merge-sort
    auto it = std::ranges::lower_bound(sortedVectorForContext, ctx.time, std::less<>{}, [](const auto& pair) { return pair.context.time; });
    sortedVectorForContext.insert(it, {ctx, newParameters});
}

void CtxSettingsBase::removeExpiredStoredParameters() {
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

// --- Private helpers: tag parsing ---

std::optional<std::string> CtxSettingsBase::contextInTag(const property_map_view& tagMap) const {
    for (const auto& [key, value] : tagMap) {
        if (gr::tag::settingsKey(key) == gr::tag::CONTEXT.shortKey()) {
            const auto context = value.value_or(std::string_view{});
            return context.data() == nullptr ? std::nullopt : std::optional<std::string>(context);
        }
    }
    return std::nullopt;
}

std::optional<std::uint64_t> CtxSettingsBase::contextTimeInTag(const property_map_view& tagMap) const {
    std::optional<std::uint64_t> contextTime;
    for (const auto& [key, value] : tagMap) {
        const std::string_view stripped = gr::tag::settingsKey(key);
        if (stripped == gr::tag::TRIGGER_TIME.shortKey()) {
            const auto time = pmt::convert_safely<std::uint64_t>(value);
            return time ? std::optional<std::uint64_t>(*time) : std::nullopt;
        }
        if (stripped == gr::tag::CONTEXT_TIME.shortKey()) {
            const auto time = pmt::convert_safely<std::uint64_t>(value);
            contextTime     = time ? std::optional<std::uint64_t>(*time) : std::nullopt;
        }
    }
    return contextTime;
}

std::optional<SettingsCtx> CtxSettingsBase::createSettingsCtxFromTag(const property_map_view& tagMap) const {
    // If CONTEXT is not present then return std::nullopt

    if (auto ctxValue = contextInTag(tagMap); ctxValue.has_value()) {
        SettingsCtx ctx{};
        ctx.context = ctxValue.value();

        if (auto contextTime = contextTimeInTag(tagMap); contextTime.has_value()) {
            ctx.time = contextTime.value();
        }
        if (ctx.time == 0ULL) {
            ctx.time = settings::convertTimePointToUint64Ns(std::chrono::system_clock::now());
        }
        return ctx;
    } else {
        return std::nullopt;
    }
}

} // namespace gr
