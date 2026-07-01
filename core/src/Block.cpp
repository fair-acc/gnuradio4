#include <gnuradio-4.0/Block.hpp>

namespace gr {

void BlockBase::initStandardPropertyCallbacks() noexcept {
    auto* alloc = propertyCallbacks.get_allocator().resource();
    auto  add   = [&](std::string_view key, PropertyCallback cb) { propertyCallbacks.try_emplace(std::pmr::string(key, alloc), cb); };
    add(block::property::kHeartbeat, &BlockBase::propertyCallbackHeartbeat);
    add(block::property::kEcho, &BlockBase::propertyCallbackEcho);
    add(block::property::kLifeCycleState, &BlockBase::propertyCallbackLifecycleState);
    add(block::property::kSetting, &BlockBase::propertyCallbackSettings);
    add(block::property::kStagedSetting, &BlockBase::propertyCallbackStagedSettings);
    add(block::property::kStoreDefaults, &BlockBase::propertyCallbackStoreDefaults);
    add(block::property::kResetDefaults, &BlockBase::propertyCallbackResetDefaults);
    add(block::property::kActiveContext, &BlockBase::propertyCallbackActiveContext);
    add(block::property::kSettingsCtx, &BlockBase::propertyCallbackSettingsCtx);
    add(block::property::kSettingsContexts, &BlockBase::propertyCallbackSettingsContexts);
    add(block::property::kMetaInformation, &BlockBase::propertyCallbackMetaInformation);
    add(block::property::kUiConstraints, &BlockBase::propertyCallbackUiConstraints);
}

std::optional<Message> BlockBase::propertyCallbackHeartbeat(std::string_view propertyName, Message message) {
    using enum gr::message::Command;
    assert(propertyName == block::property::kHeartbeat);

    if (message.cmd == Set || message.cmd == Get) {
        std::uint64_t nanoseconds_count = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count());
        message.data                    = property_map{{"heartbeat", nanoseconds_count}};
        return message;
    } else if (message.cmd == Subscribe) {
        if (!message.clientRequestID.empty()) {
            propertySubscriptions[std::pmr::string(propertyName, propertySubscriptions.get_allocator())].insert(std::pmr::string(message.clientRequestID, propertySubscriptions.get_allocator()));
        }
        return std::nullopt;
    } else if (message.cmd == Unsubscribe) {
        propertySubscriptions[std::pmr::string(propertyName, propertySubscriptions.get_allocator())].erase(std::pmr::string(message.clientRequestID, propertySubscriptions.get_allocator()));
        return std::nullopt;
    }

    throw gr::exception(std::format("block {} property {} does not implement command {}, msg: {}", cbUniqueName(), propertyName, message.cmd, message));
}

std::optional<Message> BlockBase::propertyCallbackEcho(std::string_view propertyName, Message message) {
    using enum gr::message::Command;
    assert(propertyName == block::property::kEcho);

    if (message.cmd == Set) {
        return message; // mirror message as is
    }

    throw gr::exception(std::format("block {} property {} does not implement command {}, msg: {}", cbUniqueName(), propertyName, message.cmd, message));
}

std::optional<Message> BlockBase::propertyCallbackLifecycleState(std::string_view propertyName, Message message) {
    using enum gr::message::Command;
    assert(propertyName == block::property::kLifeCycleState);

    if (message.cmd == Set) {
        if (!message.data.has_value() || !message.data.value().contains("state")) {
            throw gr::exception(std::format("propertyCallbackLifecycleState - cannot set block state w/o 'state' data msg: {}", message));
        }

        const auto& dataMap = message.data.value();
        auto        it      = dataMap.find("state");
        if (it == dataMap.end()) {
            throw gr::exception(std::format("propertyCallbackLifecycleState - state not found, msg: {}", message));
        }

        const Value stateEntry = (*it).second; // bind to lvalue; ValueMap iter yields by value
        const auto  stateStr   = stateEntry.value_or(std::string_view{});
        if (!stateStr.data()) {
            throw gr::exception(std::format("propertyCallbackLifecycleState - state is not a string, msg: {}", message));
        }

        auto state = gr::meta::parseEnum<lifecycle::State>(stateStr);
        if (!state.has_value()) {
            throw gr::exception(std::format("propertyCallbackLifecycleState - invalid lifecycle::State conversion from {}, msg: {}", stateStr, message));
        }

        if (auto e = cbChangeStateTo(state.value()); !e) {
            throw gr::exception(std::format("propertyCallbackLifecycleState - error in state transition - what: {}", e.error().message, e.error().sourceLocation, e.error().errorTime));
        }

        return std::nullopt;
    }

    if (message.cmd == Get) {
        message.data = property_map{{"state", std::string(gr::meta::enumName(cbState()).value_or(""))}};
        return message;
    }

    if (message.cmd == Subscribe) {
        if (!message.clientRequestID.empty()) {
            propertySubscriptions[std::pmr::string(propertyName, propertySubscriptions.get_allocator())].insert(std::pmr::string(message.clientRequestID, propertySubscriptions.get_allocator()));
        }
        return std::nullopt;
    }

    if (message.cmd == Unsubscribe) {
        propertySubscriptions[std::pmr::string(propertyName, propertySubscriptions.get_allocator())].erase(std::pmr::string(message.clientRequestID, propertySubscriptions.get_allocator()));
        return std::nullopt;
    }

    throw gr::exception(std::format("propertyCallbackLifecycleState - does not implement command {}, msg: {}", message.cmd, message));
}

std::optional<Message> BlockBase::propertyCallbackSettings(std::string_view propertyName, Message message) {
    using enum gr::message::Command;
    assert(propertyName == block::property::kSetting);

    if (message.cmd == Set) {
        if (!message.data.has_value()) {
            throw gr::exception(std::format("block {} (aka. {}) cannot set {} w/o data msg: {}", cbUniqueName(), cbName(), propertyName, message));
        }
        // delegate to 'propertyCallbackStagedSettings' since we cannot set but only stage new settings due to mandatory real-time/non-real-time decoupling
        // settings are applied during the next work(...) invocation.
        propertyCallbackStagedSettings(block::property::kStagedSetting, message);
        return std::nullopt;
    } else if (message.cmd == Get) {
        message.data = cbSettings().get();
        return message;
    } else if (message.cmd == Subscribe) {
        if (!message.clientRequestID.empty()) {
            propertySubscriptions[std::pmr::string(propertyName, propertySubscriptions.get_allocator())].insert(std::pmr::string(message.clientRequestID, propertySubscriptions.get_allocator()));
        }
        return std::nullopt;
    } else if (message.cmd == Unsubscribe) {
        propertySubscriptions[std::pmr::string(propertyName, propertySubscriptions.get_allocator())].erase(std::pmr::string(message.clientRequestID, propertySubscriptions.get_allocator()));
        return std::nullopt;
    }

    throw gr::exception(std::format("block {} property {} does not implement command {}, msg: {}", cbUniqueName(), propertyName, message.cmd, message));
}

std::optional<Message> BlockBase::propertyCallbackStagedSettings(std::string_view propertyName, Message message) {
    using enum gr::message::Command;
    assert(propertyName == block::property::kStagedSetting);
    const auto keys = [](const property_map& map) noexcept {
        std::string result;
        for (const auto& pair : map) {
            if (!result.empty()) {
                result += ", ";
            }
            result += pair.first;
        }
        return result;
    };

    if (message.cmd == Set) {
        if (!message.data.has_value()) {
            throw gr::exception(std::format("block {} (aka. {}) cannot set {} w/o data msg: {}", cbUniqueName(), cbName(), propertyName, message));
        }

        property_map notSet          = cbSettings().setStaged(*message.data);
        property_map stagedParameter = cbSettings().stagedParameters();

        if (notSet.empty()) {
            if (!message.clientRequestID.empty()) {
                message.cmd  = Final;
                message.data = std::move(stagedParameter);
                return message;
            }
            return std::nullopt;
        }

        throw gr::exception(std::format("propertyCallbackStagedSettings - could not set fields: {}\nvs. available: {}", keys(std::move(notSet)), keys(cbSettings().get())));
    } else if (message.cmd == Get) {
        message.data = cbSettings().stagedParameters();
        return message;
    } else if (message.cmd == Subscribe) {
        if (!message.clientRequestID.empty()) {
            propertySubscriptions[std::pmr::string(propertyName, propertySubscriptions.get_allocator())].insert(std::pmr::string(message.clientRequestID, propertySubscriptions.get_allocator()));
        }
        return std::nullopt;
    } else if (message.cmd == Unsubscribe) {
        propertySubscriptions[std::pmr::string(propertyName, propertySubscriptions.get_allocator())].erase(std::pmr::string(message.clientRequestID, propertySubscriptions.get_allocator()));
        return std::nullopt;
    }

    throw gr::exception(std::format("block {} property {} does not implement command {}, msg: {}", cbUniqueName(), propertyName, message.cmd, message));
}

std::optional<Message> BlockBase::propertyCallbackStoreDefaults(std::string_view propertyName, Message message) {
    using enum gr::message::Command;
    assert(propertyName == block::property::kStoreDefaults);

    if (message.cmd == Set) {
        cbSettings().storeDefaults();
        return std::nullopt;
    }

    throw gr::exception(std::format("block {} property {} does not implement command {}, msg: {}", cbUniqueName(), propertyName, message.cmd, message));
}

std::optional<Message> BlockBase::propertyCallbackResetDefaults(std::string_view propertyName, Message message) {
    using enum gr::message::Command;
    assert(propertyName == block::property::kResetDefaults);

    if (message.cmd == Set) {
        cbSettings().resetDefaults();
        return std::nullopt;
    }

    throw gr::exception(std::format("block {} property {} does not implement command {}, msg: {}", cbUniqueName(), propertyName, message.cmd, message));
}

std::optional<Message> BlockBase::propertyCallbackActiveContext(std::string_view propertyName, Message message) {
    using enum gr::message::Command;
    assert(propertyName == block::property::kActiveContext);

    if (message.cmd == Set) {
        if (!message.data.has_value()) {
            throw gr::exception(std::format("block {} (aka. {}) cannot set {} w/o data msg: {}", cbUniqueName(), cbName(), propertyName, message));
        }

        const auto& dataMap = message.data.value();

        std::string contextStr;
        if (auto ctxIt = dataMap.find(std::string_view{gr::tag::CONTEXT}); ctxIt != dataMap.end()) {
            const Value ctxEntry = (*ctxIt).second; // bind to lvalue; ValueMap iter yields by value
            if (const auto str = ctxEntry.value_or(std::string_view{}); str.data()) {
                contextStr = str;
            } else {
                throw gr::exception(std::format("propertyCallbackActiveContext - context is not a string, msg: {}", message));
            }
        } else if (auto ctxShortIt = dataMap.find(gr::tag::CONTEXT.shortKey()); ctxShortIt != dataMap.end()) {
            const Value ctxEntry = (*ctxShortIt).second; // bind to lvalue; ValueMap iter yields by value
            if (const auto str = ctxEntry.value_or(std::string_view{}); str.data()) {
                contextStr = str;
            } else {
                throw gr::exception(std::format("propertyCallbackActiveContext - context is not a string, msg: {}", message));
            }
        } else {
            throw gr::exception(std::format("propertyCallbackActiveContext - context name not found, msg: {}", message));
        }

        std::uint64_t time = 0;
        if (auto timeIt = dataMap.find(std::string_view{gr::tag::CONTEXT_TIME}); timeIt != dataMap.end()) {
            const Value timeEntry = (*timeIt).second;
            if (const std::uint64_t* timePtr = timeEntry.get_if<std::uint64_t>(); timePtr) {
                time = *timePtr;
            }
        } else if (auto timeShortIt = dataMap.find(gr::tag::CONTEXT_TIME.shortKey()); timeShortIt != dataMap.end()) {
            const Value timeEntry = (*timeShortIt).second;
            if (const std::uint64_t* timePtr = timeEntry.get_if<std::uint64_t>(); timePtr) {
                time = *timePtr;
            }
        }

        auto ctx = cbSettings().activateContext(SettingsCtx{
            .time    = time,
            .context = contextStr,
        });

        if (!ctx.has_value()) {
            throw gr::exception(std::format("propertyCallbackActiveContext - failed to activate context {}, msg: {}", contextStr, message));
        }
    }

        if (message.cmd == Get || message.cmd == Set) {
        const auto& ctx = cbSettings().activeContext();
        message.data = property_map{
            {std::string_view{gr::tag::CONTEXT}, ctx.context},  //
            {std::string_view{gr::tag::CONTEXT_TIME}, ctx.time} //
        };
        return message;
    }

    throw gr::exception(std::format("block {} property {} does not implement command {}, msg: {}", cbUniqueName(), propertyName, message.cmd, message));
}

std::optional<Message> BlockBase::propertyCallbackSettingsCtx(std::string_view propertyName, Message message) {
    using enum gr::message::Command;
    assert(propertyName == block::property::kSettingsCtx);

    if (!message.data.has_value()) {
        throw gr::exception(std::format("block {} (aka. {}) cannot get/set {} w/o data msg: {}", cbUniqueName(), cbName(), propertyName, message));
    }

    const auto& dataMap = message.data.value();

    std::string contextStr;
    if (auto ctxIt = dataMap.find(std::string_view{gr::tag::CONTEXT}); ctxIt != dataMap.end()) {
        const Value ctxEntry = (*ctxIt).second; // bind to lvalue; ValueMap iter yields by value
        if (const auto str = ctxEntry.value_or(std::string_view{}); str.data()) {
            contextStr = str;
        } else {
            throw gr::exception(std::format("propertyCallbackSettingsCtx - context is not a string, msg: {}", message));
        }
    } else if (auto ctxShortIt = dataMap.find(gr::tag::CONTEXT.shortKey()); ctxShortIt != dataMap.end()) {
        const Value ctxEntry = (*ctxShortIt).second; // bind to lvalue; ValueMap iter yields by value
        if (const auto str = ctxEntry.value_or(std::string_view{}); str.data()) {
            contextStr = str;
        } else {
            throw gr::exception(std::format("propertyCallbackSettingsCtx - context is not a string, msg: {}", message));
        }
    } else {
        throw gr::exception(std::format("propertyCallbackSettingsCtx - context name not found, msg: {}", message));
    }

    std::uint64_t time = 0;
    if (auto timeIt = dataMap.find(std::string_view{gr::tag::CONTEXT_TIME}); timeIt != dataMap.end()) {
        const Value timeEntry = (*timeIt).second; // bind to lvalue; ValueMap iter yields by value
        if (const std::uint64_t* timePtr = timeEntry.get_if<std::uint64_t>(); timePtr) {
            time = *timePtr;
        }
    } else if (auto timeShortIt = dataMap.find(gr::tag::CONTEXT_TIME.shortKey()); timeShortIt != dataMap.end()) {
        const Value timeEntry = (*timeShortIt).second; // bind to lvalue; ValueMap iter yields by value
        if (const std::uint64_t* timePtr = timeEntry.get_if<std::uint64_t>(); timePtr) {
            time = *timePtr;
        }
    }

    SettingsCtx ctx{
        .time    = time,
        .context = contextStr,
    };

    property_map parameters;
    if (message.cmd == Get) {
        Tensor<Value> paramKeys;
        auto          itParam = dataMap.find("parameters");
        if (itParam != dataMap.end()) {
            const Value keysEntry = (*itParam).second; // bind to lvalue; ValueMap iter yields by value
            if (auto keys = keysEntry.get_if<TensorView<Value>>()) {
                paramKeys = keys->owned();
            } else {
                std::println("Warning: keys are not Tensor<Value>");
            }
        }

        auto paramKeyStrings =                                                                                                                         //
            paramKeys | std::views::transform([](const auto& keyValue) { return keyValue.value_or(std::string()); }) | std::ranges::to<std::vector>(); //
        if (auto params = cbSettings().getStored(paramKeyStrings, ctx); params.has_value()) {
            parameters = params.value();
        }
        message.data = property_map{{"parameters", parameters}};
        return message;
    }

    if (message.cmd == Set) {
        if (auto it = dataMap.find("parameters"); it != dataMap.end()) {
            const Value paramsEntry = (*it).second; // bind to lvalue; Map* aliases entry's _storage
            auto        params      = paramsEntry.get_if<property_map>();
            if (params) {
                parameters = *params;
            }
        }

        message.data = property_map{{"failed_to_set", cbSettings().set(parameters, ctx)}};
        return message;
    }

    // Removed a Context
    if (message.cmd == Disconnect) {
        if (ctx.context.empty()) {
            throw gr::exception(std::format("propertyCallbackSettingsCtx - cannot delete default context, msg: {}", message));
        }

        if (!cbSettings().removeContext(ctx)) {
            throw gr::exception(std::format("propertyCallbackSettingsCtx - could not delete context {}, msg: {}", ctx.context, message));
        }
        return message;
    }

    throw gr::exception(std::format("block {} property {} does not implement command {}, msg: {}", cbUniqueName(), propertyName, message.cmd, message));
}

std::optional<Message> BlockBase::propertyCallbackSettingsContexts(std::string_view propertyName, Message message) {
    using enum gr::message::Command;
    assert(propertyName == block::property::kSettingsContexts);

    if (message.cmd == Get) {
        const auto& stored = cbSettings().getStoredAll();

        Tensor<Value>         contexts;
        Tensor<std::uint64_t> times;
        for (const auto& [ctxName, ctxParameters] : stored) {
            for (const auto& [ctx, properties] : ctxParameters) {
                contexts.push_back(ctx.context);
                times.push_back(ctx.time);
            }
        }

        message.data = property_map{
            {"contexts", std::move(contexts)},
            {"times", std::move(times)},
        };
        return message;
    }

    throw gr::exception(std::format("block {} property {} does not implement command {}, msg: {}", cbUniqueName(), propertyName, message.cmd, message));
}

std::optional<Message> BlockBase::propertyCallbackMetaInformation(std::string_view propertyName, Message message) {
    using enum gr::message::Command;
    assert(propertyName == block::property::kMetaInformation);

    if (message.cmd == Set) {
        throw gr::exception(std::format("block {} property {} does not implement command {}, msg: {}", cbUniqueName(), propertyName, message.cmd, message));
        return std::nullopt;
    } else if (message.cmd == Get) {
        message.data = cbMetaInformation(); // get
        return message;
    } else if (message.cmd == Subscribe) {
        if (!message.clientRequestID.empty()) {
            propertySubscriptions[std::pmr::string(propertyName, propertySubscriptions.get_allocator())].insert(std::pmr::string(message.clientRequestID, propertySubscriptions.get_allocator()));
        }
        return std::nullopt;
    } else if (message.cmd == Unsubscribe) {
        propertySubscriptions[std::pmr::string(propertyName, propertySubscriptions.get_allocator())].erase(std::pmr::string(message.clientRequestID, propertySubscriptions.get_allocator()));
        return std::nullopt;
    }

    throw gr::exception(std::format("block {} property {} does not implement command {}, msg: {}", cbUniqueName(), propertyName, message.cmd, message));
}

std::optional<Message> BlockBase::propertyCallbackUiConstraints(std::string_view propertyName, Message message) {
    using enum gr::message::Command;
    assert(propertyName == block::property::kUiConstraints);

    if (message.cmd == Set) {
        if (!message.data.has_value()) {
            throw gr::exception(std::format("block {} (aka. {}) cannot set {} w/o data msg: {}", cbUniqueName(), cbName(), propertyName, message));
        }
        // delegate to 'propertyCallbackStagedSettings' since we cannot set but only stage new settings due to mandatory real-time/non-real-time decoupling
        // settings are applied during the next work(...) invocation.
        propertyCallbackStagedSettings(block::property::kStagedSetting, message);
        return std::nullopt;
    } else if (message.cmd == Get) {      // only return ui_constraints
        message.data = cbUiConstraints(); // get
        return message;
    } else if (message.cmd == Subscribe) {
        if (!message.clientRequestID.empty()) {
            propertySubscriptions[std::pmr::string(propertyName, propertySubscriptions.get_allocator())].insert(std::pmr::string(message.clientRequestID, propertySubscriptions.get_allocator()));
        }
        return std::nullopt;
    } else if (message.cmd == Unsubscribe) {
        propertySubscriptions[std::pmr::string(propertyName, propertySubscriptions.get_allocator())].erase(std::pmr::string(message.clientRequestID, propertySubscriptions.get_allocator()));
        return std::nullopt;
    }

    throw gr::exception(std::format("block {} property {} does not implement command {}, msg: {}", cbUniqueName(), propertyName, message.cmd, message));
}

std::optional<Message> BlockBase::propertyCallbackSubgraphExport([[maybe_unused]] std::string_view propertyName, Message message) {
    if (_subgraphExportHandler != nullptr) {
        return _subgraphExportHandler(_subgraphExportContext, std::move(message));
    }
    return std::nullopt;
}

} // namespace gr
