#include <gnuradio-4.0/Block.hpp>

namespace gr {

std::optional<Message> BlockBase::propertyCallbackHeartbeat(std::string_view propertyName, Message message) {
    using enum gr::message::Command;
    assert(propertyName == block::property::kHeartbeat);

    if (message.cmd == Set || message.cmd == Get) {
        std::uint64_t nanoseconds_count = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count());
        message.data                    = pmt::Value::Map{{"heartbeat", nanoseconds_count}};
        return message;
    } else if (message.cmd == Subscribe) {
        if (!message.clientRequestID.empty()) {
            propertySubscriptions[std::string(propertyName)].insert(message.clientRequestID);
        }
        return std::nullopt;
    } else if (message.cmd == Unsubscribe) {
        propertySubscriptions[std::string(propertyName)].erase(message.clientRequestID);
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

        const auto stateStr = it->second.value_or(std::string_view{});
        if (!stateStr.data()) {
            throw gr::exception(std::format("propertyCallbackLifecycleState - state is not a string, msg: {}", message));
        }

        auto state = magic_enum::enum_cast<lifecycle::State>(stateStr);
        if (!state.has_value()) {
            throw gr::exception(std::format("propertyCallbackLifecycleState - invalid lifecycle::State conversion from {}, msg: {}", stateStr, message));
        }

        if (auto e = cbChangeStateTo(state.value()); !e) {
            throw gr::exception(std::format("propertyCallbackLifecycleState - error in state transition - what: {}", e.error().message, e.error().sourceLocation, e.error().errorTime));
        }

        return std::nullopt;
    }

    if (message.cmd == Get) {
        message.data = pmt::Value::Map{{"state", std::string(magic_enum::enum_name(cbState()))}};
        return message;
    }

    if (message.cmd == Subscribe) {
        if (!message.clientRequestID.empty()) {
            propertySubscriptions[std::string(propertyName)].insert(message.clientRequestID);
        }
        return std::nullopt;
    }

    if (message.cmd == Unsubscribe) {
        propertySubscriptions[std::string(propertyName)].erase(message.clientRequestID);
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
            propertySubscriptions[std::string(propertyName)].insert(message.clientRequestID);
        }
        return std::nullopt;
    } else if (message.cmd == Unsubscribe) {
        propertySubscriptions[std::string(propertyName)].erase(message.clientRequestID);
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
            propertySubscriptions[std::string(propertyName)].insert(message.clientRequestID);
        }
        return std::nullopt;
    } else if (message.cmd == Unsubscribe) {
        propertySubscriptions[std::string(propertyName)].erase(message.clientRequestID);
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
        if (auto it = dataMap.find(gr::tag::CONTEXT.shortKey()); it != dataMap.end()) {
            if (const auto str = it->second.value_or(std::string_view{}); str.data()) {
                contextStr = str;
            } else {
                throw gr::exception(std::format("propertyCallbackActiveContext - context is not a string, msg: {}", message));
            }
        } else {
            throw gr::exception(std::format("propertyCallbackActiveContext - context name not found, msg: {}", message));
        }

        std::uint64_t time = 0;
        if (auto it = dataMap.find(gr::tag::CONTEXT_TIME.shortKey()); it != dataMap.end()) {
            if (const std::uint64_t* timePtr = it->second.get_if<std::uint64_t>(); timePtr) {
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
        message.data    = property_map{
               {gr::tag::CONTEXT.shortKey(), ctx.context},  //
               {gr::tag::CONTEXT_TIME.shortKey(), ctx.time} //
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
    if (auto it = dataMap.find(gr::tag::CONTEXT.shortKey()); it != dataMap.end()) {
        if (const auto str = it->second.value_or(std::string_view{}); str.data()) {
            contextStr = str;
        } else {
            throw gr::exception(std::format("propertyCallbackSettingsCtx - context is not a string, msg: {}", message));
        }
    } else {
        throw gr::exception(std::format("propertyCallbackSettingsCtx - context name not found, msg: {}", message));
    }

    std::uint64_t time = 0;
    if (auto it = dataMap.find(gr::tag::CONTEXT_TIME.shortKey()); it != dataMap.end()) {
        if (const std::uint64_t* timePtr = it->second.get_if<std::uint64_t>(); timePtr) {
            time = *timePtr;
        }
    }

    SettingsCtx ctx{
        .time    = time,
        .context = contextStr,
    };

    pmt::Value::Map parameters;
    if (message.cmd == Get) {
        Tensor<pmt::Value> paramKeys;
        auto               itParam = dataMap.find("parameters");
        if (itParam != dataMap.end()) {
            auto keys = itParam->second.get_if<Tensor<pmt::Value>>();
            if (keys) {
                paramKeys = *keys;
            } else {
                std::println("Warning: keys are not Tensor<Value>");
            }
        }

        auto paramKeyStrings =                                                                                                                         //
            paramKeys | std::views::transform([](const auto& keyValue) { return keyValue.value_or(std::string()); }) | std::ranges::to<std::vector>(); //
        if (auto params = cbSettings().getStored(paramKeyStrings, ctx); params.has_value()) {
            parameters = params.value();
        }
        message.data = pmt::Value::Map{{"parameters", parameters}};
        return message;
    }

    if (message.cmd == Set) {
        if (auto it = dataMap.find("parameters"); it != dataMap.end()) {
            auto params = it->second.get_if<pmt::Value::Map>();
            if (params) {
                parameters = *params;
            }
        }

        message.data = property_map{{"failed_to_set", cbSettings().set(parameters, ctx)}};
        return message;
    }

    // Removed a Context
    if (message.cmd == Disconnect) {
        auto str = ctx.context.value_or(std::string_view{});
        if (str.empty()) {
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
        const std::map<pmt::Value, std::vector<SettingsBase::CtxSettingsPair>, settings::PMTCompare>& stored = cbSettings().getStoredAll();

        Tensor<pmt::Value>    contexts;
        Tensor<std::uint64_t> times;
        for (const auto& [ctxName, ctxParameters] : stored) {
            for (const auto& [ctx, properties] : ctxParameters) {
                if (!ctx.context.holds<std::string>()) {
                    continue;
                }
                const auto str = ctx.context.value_or(std::string_view{});
                contexts.push_back(str);
                times.push_back(ctx.time);
            }
        }

        message.data = pmt::Value::Map{
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
            propertySubscriptions[std::string(propertyName)].insert(message.clientRequestID);
        }
        return std::nullopt;
    } else if (message.cmd == Unsubscribe) {
        propertySubscriptions[std::string(propertyName)].erase(message.clientRequestID);
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
            propertySubscriptions[std::string(propertyName)].insert(message.clientRequestID);
        }
        return std::nullopt;
    } else if (message.cmd == Unsubscribe) {
        propertySubscriptions[std::string(propertyName)].erase(message.clientRequestID);
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
