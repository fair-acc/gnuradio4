#include <gnuradio-4.0/Settings.hpp>

namespace gr {

// --- Simple accessors ---

bool CtxSettingsBase::changed() const noexcept { return gr::atomic_ref(_changed).load_acquire(); }

void CtxSettingsBase::setChanged(bool b) noexcept { gr::atomic_ref(_changed).store_release(b); }

void CtxSettingsBase::setInitBlockParameters(const property_map& parameters) { _initBlockParameters = parameters; }

const SettingsCtx& CtxSettingsBase::activeContext() const noexcept { return _activeCtx; }

std::set<std::string>& CtxSettingsBase::autoForwardParameters() noexcept { return _autoForwardParameters; }

const property_map& CtxSettingsBase::defaultParameters() const noexcept { return _defaultParameters; }

const property_map& CtxSettingsBase::activeParameters() const noexcept { return _activeParameters; }

// --- get() overloads ---

property_map CtxSettingsBase::get(std::span<const std::string> parameterKeys) const noexcept {
    std::lock_guard lg(_mutex);
    if (parameterKeys.empty()) {
        return _activeParameters;
    }
    property_map ret;
    for (const auto& key : parameterKeys) {
        if (_activeParameters.contains(convert_string_domain(key))) {
            ret.insert_or_assign(convert_string_domain(key), _activeParameters.at(convert_string_domain(key)));
        }
    }
    return ret;
}

std::optional<pmt::Value> CtxSettingsBase::get(const std::string& parameterKey) const noexcept {
    auto res = get(std::array<std::string, 1>({parameterKey}));
    auto it  = res.find(convert_string_domain(parameterKey));
    if (it != res.end()) {
        return it->second;
    } else {
        return std::nullopt;
    }
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
    property_map ret;
    for (const auto& key : parameterKeys) {
        if (allBestMatchParameters->contains(convert_string_domain(key))) {
            ret.insert_or_assign(convert_string_domain(key), allBestMatchParameters->at(convert_string_domain(key)));
        }
    }
    return ret;
}

std::optional<pmt::Value> CtxSettingsBase::getStored(const std::string& parameterKey, SettingsCtx ctx) const noexcept {
    auto res = getStored(std::array<std::string, 1>({parameterKey}), ctx);

    if (res.has_value() && res->contains(convert_string_domain(parameterKey))) {
        return res->at(convert_string_domain(parameterKey));
    } else {
        return std::nullopt;
    }
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

std::map<pmt::Value, std::vector<SettingsBase::CtxSettingsPair>, settings::PMTCompare> CtxSettingsBase::getStoredAll() const noexcept { return _storedParameters; }

const property_map& CtxSettingsBase::stagedParameters() const {
    std::lock_guard lg(_mutex);
    return _stagedParameters;
}

std::set<std::string> CtxSettingsBase::autoUpdateParameters(SettingsCtx ctx) noexcept {
    auto bestMatchSettingsCtx = findBestMatchSettingsCtx(ctx);
    return bestMatchSettingsCtx == std::nullopt ? std::set<std::string>() : _autoUpdateParameters[bestMatchSettingsCtx.value()];
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
            const std::set<std::string>& currentAutoUpdateParams = _autoUpdateParameters.at(bestMatchSettingsCtx.value());

            // the following is more compile-time friendly
            property_map notAutoUpdateParams;
            for (const auto& pair : parameters.value()) {
                if (!currentAutoUpdateParams.contains(std::string(pair.first))) {
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
    auto str = ctx.context.value_or(std::string_view{});
    if (str.empty()) {
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

// --- assignFrom ---

void CtxSettingsBase::assignFrom(const CtxSettingsBase& other) {
    std::scoped_lock lock(_mutex, other._mutex);
    gr::atomic_ref(_changed).store_release(gr::atomic_ref(other._changed).load_acquire());
    _storedParameters      = other._storedParameters;
    _defaultParameters     = other._defaultParameters;
    _initBlockParameters   = other._initBlockParameters;
    _autoUpdateParameters  = other._autoUpdateParameters;
    _autoForwardParameters = other._autoForwardParameters;
    _matchPred             = other._matchPred;
    _activeCtx             = other._activeCtx;
    _stagedParameters      = other._stagedParameters;
    _activeParameters      = other._activeParameters;
}

void CtxSettingsBase::assignFrom(CtxSettingsBase&& other) noexcept {
    std::scoped_lock lock(_mutex, other._mutex);
    gr::atomic_ref(_changed).store_release(gr::atomic_ref(other._changed).load_acquire());
    _storedParameters      = std::move(other._storedParameters);
    _defaultParameters     = std::move(other._defaultParameters);
    _initBlockParameters   = std::move(other._initBlockParameters);
    _autoUpdateParameters  = std::move(other._autoUpdateParameters);
    _autoForwardParameters = std::move(other._autoForwardParameters);
    _matchPred             = std::exchange(other._matchPred, settings::nullMatchPred);
    _activeCtx             = std::exchange(other._activeCtx, {});
    _stagedParameters      = std::move(other._stagedParameters);
    _activeParameters      = std::move(other._activeParameters);
}

// --- Private helpers: match/search ---

std::optional<pmt::Value> CtxSettingsBase::findBestMatchCtx(const pmt::Value& contextToSearch) const {
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

std::optional<SettingsCtx> CtxSettingsBase::findBestMatchSettingsCtx(const SettingsCtx& ctx) const {
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

std::optional<property_map> CtxSettingsBase::getBestMatchStoredParameters(const SettingsCtx& ctx) const {
    const auto bestMatchSettingsCtx = findBestMatchSettingsCtx(ctx);
    if (bestMatchSettingsCtx == std::nullopt) {
        return std::nullopt;
    }
    const auto& vec        = _storedParameters[bestMatchSettingsCtx.value().context];
    const auto  parameters = std::ranges::find_if(vec, [&](const CtxSettingsPair& contextSettings) { return contextSettings.context == bestMatchSettingsCtx.value(); });

    return parameters != vec.end() ? std::optional(parameters->settings) : std::nullopt;
}

std::optional<std::set<std::string>> CtxSettingsBase::getBestMatchAutoUpdateParameters(const SettingsCtx& ctx) const {
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
        _autoUpdateParameters[ctx] = getBestMatchAutoUpdateParameters(ctx).value_or(doGetAllWritableMembers());
    }

    std::vector<CtxSettingsPair>& sortedVectorForContext = _storedParameters[ctx.context];
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

std::optional<std::string> CtxSettingsBase::contextInTag(const Tag& tag) const {
    if (tag.map.contains(gr::tag::CONTEXT.shortKey())) {
        const pmt::Value& ctxInfo = tag.map.at(gr::tag::CONTEXT.shortKey());
        auto              result  = ctxInfo.value_or(std::string_view{});
        if (result.data() != nullptr) {
            return {std::string(result)};
        }
    }
    return std::nullopt;
}

std::optional<std::uint64_t> CtxSettingsBase::triggeredTimeInTag(const Tag& tag) const {
    if (tag.map.contains(gr::tag::TRIGGER_TIME.shortKey())) {
        const pmt::Value& pmtTimeUtcNs = tag.map.at(gr::tag::TRIGGER_TIME.shortKey());
        auto              result       = pmt::convert_safely<std::uint64_t>(pmtTimeUtcNs);
        if (result) {
            return *result;
        }
    }
    return std::nullopt;
}

std::optional<SettingsCtx> CtxSettingsBase::createSettingsCtxFromTag(const Tag& tag) const {
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

} // namespace gr
