#ifndef GRAPH_PROTOTYPE_TRANSACTIONS_HPP
#define GRAPH_PROTOTYPE_TRANSACTIONS_HPP

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

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat-nonliteral"
#include <fmt/chrono.h>
#pragma GCC diagnostic pop

#include "circular_buffer.hpp"
#include "claim_strategy.hpp"
#include "reader_writer_lock.hpp"
#include "sequence.hpp"
#include "timingctx.hpp"
#include "utils.hpp"
#include "wait_strategy.hpp"

namespace fair::graph {

namespace settings {
template<typename Node>
struct alignas(fair::meta::kCacheLine) node {
    using TimeStamp               = std::chrono::system_clock::time_point;
    std::shared_ptr<Node> value      = std::make_shared<Node>();
    TimeStamp          validSince = std::chrono::system_clock::now();
    mutable TimeStamp  lastAccess = std::chrono::system_clock::now();
    node()                        = default;
    explicit node(Node &&in) : value(std::make_shared<Node>(std::forward<Node>(in))){};

    constexpr void
    touch() const noexcept {
        lastAccess = std::chrono::system_clock::now();
    }

    explicit(false) constexpr operator Node const &() const noexcept { return *value; }
};

static_assert(sizeof(node<int>) % fair::meta::kCacheLine == 0, "node size must be cache line aligned");

struct TransactionResult {
    using TimeStamp = std::chrono::system_clock::time_point;
    const bool                isCommitted;
    const TimeStamp           timeStamp;

    explicit(false) constexpr operator bool const &() const noexcept { return isCommitted; }

    explicit(false) constexpr operator TimeStamp const &() const noexcept { return timeStamp; }

    auto
    operator<=>(const TransactionResult &) const noexcept
            = default;
};

template<typename Node>
struct CtxResult {
    const TimingCtx                       timingCtx;
    const settings::node<Node>              &settingValue;

    explicit(false) constexpr             operator auto const &() const noexcept { return timingCtx; }

    explicit(false) constexpr             operator Node const &() const noexcept { return settingValue; }

    auto
    operator<=>(const CtxResult &) const noexcept
            = default;
};

} // namespace settings

template<std::equality_comparable TransactionToken>
inline static const TransactionToken NullToken = TransactionToken{};

/**
 * @brief A thread-safe settings wrapper that supports multiple stage/commit transactions and history functionality.
 *
 * Example:
 * @code
 * fair::graph::SettingBase<int,int, std::string, 16, std::chrono::seconds, 3600 * 24, 10> settings;
 *
 * auto [ok1, timeStamp1] = settings.commit(42); // store 42 in settings
 * auto [ok2, timeStamp2] = settings.commit(43); // store 43 in settings
 *
 * assert(settings.get() == 43); // get the latest value
 * assert(settings.get(timeStamp2) == 43); // get the first isCommitted value since timeStamp2
 * assert(settings.get(-1) == 42); // get the second to last value (for lib development only)
 * assert(settings.get(timeStamp1) == 42); // get the first isCommitted value since timeStamp1
 *
 * auto [ok3, timeStamp3] = settings.stage(53, "transactionToken#1"); // stage 53 in settings, N.B. setting is not yet committed
 * assert(settings.get() == 43); // get the latest value
 * auto [ok4, timeStamp4] = settings.commit("transactionToken#1"); // commit transaction
 * assert(settings.get() == 53); // get the latest value
 * @endcode
 *
 * @tparam T is the user-supplied setting type, for simple settings U and T are identical (see CtxSettings for an example where it isn't).
 * @tparam U is the internally stored setting type that may include additional meta data.
 * @tparam TransactionToken unique identifier with which to store/commit transactions.
 * @tparam N_HISTORY the maximum number of setting history
 * @tparam TimeDiff the std::chrono::duration time-base for the time-outs
 * @tparam timeOut maximum time given in units of TimeDiff after which a setting automatically expires if unused. (default: -1 -> disabled)
 * @tparam timeOutTransactions maximum time given in units of TimeDiff after which a transaction automatically expires if not being committed. (default: -1 -> disabled)
 */
template<std::movable T, std::movable U = T, std::equality_comparable TransactionToken = std::string, std::size_t N_HISTORY = 1024, typename TimeDiff = std::chrono::seconds, int timeOut = -1,
         int timeOutTransactions = -1>
    requires(std::has_single_bit(N_HISTORY) && N_HISTORY > 8)
class SettingBase {
    using TimeStamp                                                                 = std::chrono::system_clock::time_point;
    using CircularBuffer                                                            = gr::circular_buffer<settings::node<U>, std::dynamic_extent, gr::ProducerType::Multi, gr::BusySpinWaitStrategy>;
    using Sequence                                                                  = gr::Sequence;

    constexpr static std::size_t BUFFER_MARGIN                                      = 8;
    alignas(fair::meta::kCacheLine) std::shared_ptr<CircularBuffer> _circularBuffer = std::make_shared<CircularBuffer>(N_HISTORY);
    alignas(fair::meta::kCacheLine) std::shared_ptr<Sequence> _sequenceHead         = std::make_shared<Sequence>();
    alignas(fair::meta::kCacheLine) std::shared_ptr<Sequence> _sequenceTail         = std::make_shared<Sequence>(0);
    alignas(fair::meta::kCacheLine) mutable ReaderWriterLock _historyLock{};
    alignas(fair::meta::kCacheLine) mutable std::atomic<bool> _transactionListLock{ false };
    std::list<std::pair<TransactionToken, settings::node<T>>> _transactionList;
    alignas(fair::meta::kCacheLine) const std::function<U(const U &, T &&)> _onCommit;
    alignas(fair::meta::kCacheLine) std::shared_ptr<decltype(_circularBuffer->new_writer())> _writer = std::make_shared<decltype(_circularBuffer->new_writer())>(_circularBuffer->new_writer());
    alignas(fair::meta::kCacheLine) std::shared_ptr<decltype(_circularBuffer->new_reader())> _reader = std::make_shared<decltype(_circularBuffer->new_reader())>(_circularBuffer->new_reader());

public:
    using Node              = settings::node<U>;
    using TransactionResult = settings::TransactionResult;
    SettingBase() : SettingBase([](const U & /*old*/, T &&in) -> U { return static_cast<U>(std::move(in)); }){};

    template<class Fn>
        requires std::is_invocable_r_v<U, Fn &&, const U &, T &&>
    explicit SettingBase(Fn &&onCommit) : _onCommit(std::forward<decltype(onCommit)>(onCommit)) {
        auto data     = _writer->reserve_output_range(1);
        data[0].value = std::make_shared<U>(U());
        data.publish(1);
        std::ignore = _sequenceHead->incrementAndGet();
    }

    [[nodiscard]] std::size_t
    nHistory() const noexcept {
        _historyLock.scopedGuard<ReaderWriterLockType::READ>();
        return static_cast<std::size_t>(_sequenceHead->value() - _sequenceTail->value() + 1);
    }

    ReaderWriterLock &
    historyLock() noexcept {
        return _historyLock;
    }

    TransactionResult
    stage(T &&t, const TransactionToken &transactionToken = NullToken<TransactionToken>, const TimeStamp &now = std::chrono::system_clock::now()) {
        if (transactionToken.empty()) {
            const auto oldValue   = get();
            const auto isCommited = _writer->try_publish([&](auto &w) {
                w[0].value      = std::make_shared<U>(_onCommit(*oldValue.value, std::forward<T>(t)));
                w[0].validSince = now;
                w[0].lastAccess = now;
                std::ignore     = _sequenceHead->incrementAndGet();
            });
            retireExpired();

            return { isCommited, now };
        }

        bool expected = false;
        while (std::atomic_compare_exchange_strong(&_transactionListLock, &expected, true)) // spin-lock
            ;
#if not defined(__EMSCRIPTEN__) and (not defined(__clang__) or (__clang_major__ >= 16))
        if (auto it = std::ranges::find_if(_transactionList, [&transactionToken](const auto &pair) { return pair.first == transactionToken; }); it != _transactionList.end()) {
#else
        if (auto it = std::find_if(_transactionList.begin(), _transactionList.end(), [&transactionToken](const auto &pair) { return pair.first == transactionToken; }); it != _transactionList.end()) {
#endif
            it->second = settings::node<T>(std::forward<T>(t)); // update value of existing transaction
        } else {
            _transactionList.push_back(std::make_pair(transactionToken, settings::node<T>(std::forward<T>(t))));
        }
        std::atomic_store_explicit(&_transactionListLock, false, std::memory_order_release);

        retireExpired();
        return { false, now };
    }

    TransactionResult
    commit(T &&t, const TimeStamp &now = std::chrono::system_clock::now()) {
        return stage(std::move(t), NullToken<TransactionToken>, now);
    }

    TransactionResult
    commit(const TransactionToken &transactionToken, const TimeStamp &now = std::chrono::system_clock::now()) {
        bool expected  = false;
        bool submitted = false;
        while (std::atomic_compare_exchange_strong(&_transactionListLock, &expected, true)) // spin-lock
            ;

#if not defined(__EMSCRIPTEN__) and (not defined(__clang__) or (__clang_major__ >= 16))
        const auto [first, last] = std::ranges::remove_if(_transactionList, [&transactionToken, &submitted, this, &now](const auto &setting) {
            if (transactionToken == NullToken<TransactionToken> || setting.first == transactionToken) {
                this->commit(std::move(*setting.second.value), now);
                submitted = true;
                return true;
            }
            return false;
        });
#else
        const auto first = std::remove_if(_transactionList.begin(), _transactionList.end(), [&transactionToken, &submitted, this, &now](const auto &setting) {
            if (transactionToken == NullToken<TransactionToken> || setting.first == transactionToken) {
                this->commit(std::move(*setting.second.value), now);
                submitted = true;
                return true;
            }
            return false;
        });
        const auto last  = _transactionList.end();
#endif

        _transactionList.erase(first, last);
        std::atomic_store_explicit(&_transactionListLock, false, std::memory_order_release);

        return { submitted, now };
    }

    template<class Fn>
        requires std::is_invocable_r_v<U, Fn &&, const U &> bool
    modifySetting(Fn &&modFunction, const TimeStamp &now = std::chrono::system_clock::now()) {
        const auto result = _writer->try_publish([this, &modFunction, &now](auto &w) {
            const auto oldValue = get();
            w[0].value          = std::make_shared<U>(modFunction(*oldValue.value));
            w[0].validSince     = now;
            w[0].lastAccess     = now;
            std::ignore         = _sequenceHead->incrementAndGet();
        });
        retireExpired();
        return result;
    }

    std::vector<TransactionToken>
    getPendingTransactions() const {
        std::vector<TransactionToken> result;
        bool                          expected = false;
        while (std::atomic_compare_exchange_strong(&_transactionListLock, &expected, true)) // spin-lock
            ;
        result.reserve(_transactionList.size());
#if not defined(__EMSCRIPTEN__) and (not defined(__clang__) or (__clang_major__ >= 16))
        std::ranges::transform(_transactionList, std::back_inserter(result), [](const auto &setting) { return setting.first; });
#else
        std::transform(_transactionList.begin(), _transactionList.end(), std::back_inserter(result), [](const auto &setting) { return setting.first; });
#endif
        std::atomic_store_explicit(&_transactionListLock, false, std::memory_order_release);
        return result;
    }

    [[nodiscard]] Node
    get(const TimeStamp &timeStamp) const {
        auto lHead = _sequenceHead->value();
        auto guard = _historyLock.scopedGuard<ReaderWriterLockType::READ>(); // to prevent the writer/clean-up task to potentially expire a node at the tail
        auto data  = _reader->get();
        while (data[static_cast<std::size_t>(lHead)].validSince > timeStamp && lHead != _sequenceTail->value()) {
            lHead--;
        }
        auto node = data[static_cast<std::size_t>(lHead)];
        if (node.validSince > timeStamp) {
            throw std::out_of_range(fmt::format("no settings found for the given time stamp {}", timeStamp));
        }
        node.touch();
        return node; // performs thread-safe copy of immutable object
    }

    [[nodiscard]] Node
    get(const std::int64_t idx = 0) const {
        if (idx > 0) {
            throw std::out_of_range(fmt::format("index {} must be negative or zero", idx));
        }
        auto       guard   = _historyLock.scopedGuard<ReaderWriterLockType::READ>(); // to prevent the writer/clean-up task to potentially expire a node at the tail
        const auto readIdx = _sequenceHead->value() + idx;
        if (readIdx < _sequenceTail->value()) {
            throw std::out_of_range(fmt::format("no settings found for the given index {}", idx));
        }
        auto data = _reader->get();
        auto node = data[static_cast<std::size_t>(_sequenceHead->value() + idx)];
        node.touch();
        return node; // performs thread-safe copy of immutable object
    }

    bool
    retireStaged(const TransactionToken &transactionToken = NullToken<TransactionToken>) {
        bool retired  = false;
        bool expected = false;
        while (std::atomic_compare_exchange_strong(&_transactionListLock, &expected, true)) // spin-lock
            ;

#if not defined(__EMSCRIPTEN__) and (not defined(__clang__) or (__clang_major__ >= 16))
        auto [first, last] = std::ranges::remove_if(_transactionList, [&transactionToken, &retired, this](const auto &setting) {
            if (transactionToken == NullToken<TransactionToken> || setting.first == transactionToken) {
                retired = true;
                return true;
            }
            return false;
        });
#else
        auto first = std::remove_if(_transactionList.begin(), _transactionList.end(), [&transactionToken, &retired](const auto &setting) {
            if (transactionToken == NullToken<TransactionToken> || setting.first == transactionToken) {
                retired = true;
                return true;
            }
            return false;
        });
        auto last  = _transactionList.end();
#endif
        _transactionList.erase(first, last);
        std::atomic_store_explicit(&_transactionListLock, false, std::memory_order_release);

        return retired;
    }

    void
    retireExpired(const TimeStamp &now = std::chrono::system_clock::now()) {
        if (timeOutTransactions > 0) {
            // time-out old transactions
            bool expected = false;
            while (std::atomic_compare_exchange_strong(&_transactionListLock, &expected, true)) // spin-lock
                ;
#if not defined(__EMSCRIPTEN__) and (not defined(__clang__) or (__clang_major__ >= 16))
            const auto [first, last] = std::ranges::remove_if(_transactionList,
                                                              [&now, this](const auto &setting) { return setting.second.lastAccess - now + TimeDiff{ timeOutTransactions } < TimeDiff{ 0 }; });
#else
            const auto first = std::remove_if(_transactionList.begin(), _transactionList.end(),
                                              [&now](const auto &setting) { return setting.second.lastAccess - now + TimeDiff{ timeOutTransactions } < TimeDiff{ 0 }; });
            const auto last  = _transactionList.end();
#endif
            _transactionList.erase(first, last);
            std::atomic_store_explicit(&_transactionListLock, false, std::memory_order_release);
        }

        auto guard = _historyLock.scopedGuard<ReaderWriterLockType::WRITE>();
        // expire old settings based on count
        while (_sequenceTail->value() != _sequenceHead->value() && (_sequenceHead->value() - _sequenceTail->value() > static_cast<std::int64_t>(N_HISTORY - BUFFER_MARGIN))) {
            [[maybe_unused]] auto unusedTail = _sequenceTail->incrementAndGet();
        }
        // expire old settings based on time
        if constexpr (timeOut > 0) {
            auto tailPosition = _sequenceTail->value();
            auto data         = _reader->get();
            while (tailPosition != _sequenceHead->value()) {
                if (data[static_cast<std::size_t>(tailPosition)].lastAccess - now + TimeDiff{ timeOut } < TimeDiff{ 0 }) {
                    tailPosition = _sequenceTail->incrementAndGet();
                } else {
                    tailPosition++;
                }
            }
        }
    }
};

template<std::movable T, std::size_t N_HISTORY, typename TimeDiff = std::chrono::seconds, int timeOut = -1>
    requires(std::has_single_bit(N_HISTORY) && N_HISTORY > 8)
class Setting : public SettingBase<T, T, std::string, N_HISTORY, TimeDiff, timeOut> {};

template<std::movable T, std::equality_comparable TransactionToken, std::size_t N_HISTORY = 1024, typename TimeDiff = std::chrono::seconds, int timeOut = -1, int timeOutTransactions = -1>
    requires(std::has_single_bit(N_HISTORY) && N_HISTORY > 8)
class TransactionSetting : public SettingBase<T, T, TransactionToken, N_HISTORY, TimeDiff, timeOut, timeOutTransactions> {};

template<std::movable T, std::equality_comparable TransactionToken, std::size_t N_HISTORY = 1024, typename TimeDiff = std::chrono::seconds, int timeOut = -1, int timeOutTransactions = -1>
    requires(std::has_single_bit(N_HISTORY) && N_HISTORY > 8)
class CtxSetting {
    using TimeStamp = std::chrono::system_clock::time_point;
    using Setting   = std::pair<TimingCtx, T>;
    //
    SettingBase<std::pair<TimingCtx, T>, std::unordered_map<TimingCtx, settings::node<T>>, TransactionToken, N_HISTORY, TimeDiff, timeOut, timeOutTransactions> _setting{
        [](const std::unordered_map<TimingCtx, settings::node<T>> &oldMap, std::pair<TimingCtx, T> &&newValue) -> std::unordered_map<TimingCtx, settings::node<T>> {
            auto newMap = oldMap;
            if (auto it = newMap.find(newValue.first); it != newMap.end()) {
                it->second = settings::node(std::forward<decltype(newValue.second)>(newValue.second));
            } else {
                newMap.emplace(newValue.first, settings::node(std::move(newValue.second)));
            }
            return newMap;
        }
    };

public:
    using Node                     = settings::node<T>;
    using TransactionResult        = settings::TransactionResult;
    using CtxResult                = settings::CtxResult<T>;
    CtxSetting()                   = default;
    CtxSetting(const CtxSetting &) = delete;
    CtxSetting &
    operator=(const CtxSetting &)
            = delete;

    TransactionResult
    stage(const TimingCtx &timingCtx, T &&newValue, const TransactionToken &transactionToken = NullToken<TransactionToken>, const TimeStamp &now = std::chrono::system_clock::now()) {
        return _setting.stage({ timingCtx, std::forward<T>(newValue) }, transactionToken, now);
    }

    [[maybe_unused]] bool
    retireStaged(const TransactionToken &transactionToken = NullToken<TransactionToken>) {
        return _setting.retireStaged(transactionToken);
    }

    TransactionResult
    commit(const TimingCtx &timingCtx, T &&newValue, const TimeStamp &now = std::chrono::system_clock::now()) {
        return stage(timingCtx, std::forward<T>(newValue), NullToken<TransactionToken>, now);
    }

    TransactionResult
    commit(const TransactionToken &transactionToken = NullToken<TransactionToken>, const TimeStamp &now = std::chrono::system_clock::now()) {
        return _setting.commit(transactionToken, now);
    }

    [[nodiscard]] CtxResult
    get(const TimingCtx &timingCtx = TimingCtx(), const std::int64_t idx = 0) const {
        return get(*_setting.get(idx).value, timingCtx);
    }

    [[nodiscard]] CtxResult
    get(const TimingCtx &timingCtx, const TimeStamp &timeStamp) const {
        return get(*_setting.get(timeStamp).value, timingCtx);
    }

    [[nodiscard]] std::size_t
    nHistory() const {
        return _setting.nHistory();
    }

    [[nodiscard]] std::size_t
    nCtxHistory(const std::int64_t idx = 0) const {
        return _setting.get(idx).value->size();
    }

    [[nodiscard]] std::vector<TransactionToken>
    getPendingTransactions() const {
        return _setting.getPendingTransactions();
    }

    void
    retireExpired(const TimeStamp &now = std::chrono::system_clock::now()) {
        _setting.historyLock().template scopedGuard<ReaderWriterLockType::WRITE>();
        _setting.retireExpired(now);
        retireOldSettings(*_setting.get().value, now);
    }

    template<bool exactMatch = false>
    [[maybe_unused]] bool
    retire(const TimingCtx &ctx, const TimeStamp &now = std::chrono::system_clock::now()) {
        bool modifiedSettings = false;
        _setting.modifySetting(
                [&ctx, &modifiedSettings](const std::unordered_map<TimingCtx, settings::node<T>> &oldSetting) {
                    auto newSetting = oldSetting;
                    if constexpr (exactMatch) {
                        modifiedSettings = std::erase_if(newSetting, [&ctx](const std::pair<TimingCtx, settings::node<T>> &pair) { return pair.first == ctx; });
                    } else {
                        modifiedSettings = std::erase_if(newSetting, [&ctx](const auto &pair) { return pair.first.matches(ctx); });
                    }
                    return newSetting;
                },
                now);
        return modifiedSettings;
    }

private:
    [[nodiscard]] CtxResult
    get(const auto &settingsMap, const TimingCtx &timingCtx) const noexcept {
        // prefer exact match
        for (const auto &[key, value] : settingsMap) {
            if (key == timingCtx) {
                value.touch();
                return { timingCtx, value };
            }
        }

        // no exact match, but maybe something matches
        for (const auto &[key, value] : settingsMap) {
            if (key.matches(timingCtx) && key != NullTimingCtx) {
                value.touch();
                return { timingCtx, value };
            }
        }

        // did not find a match the setting for the specific timing context
        return CtxResult(TimingCtx(), settings::node<T>());
    }

    void
    retireOldSettings(auto &settingsMap, const TimeStamp &now = std::chrono::system_clock::now()) const noexcept {
        for (auto it = settingsMap.begin(); it != settingsMap.end();) {
            if (it->second.lastAccess - now + TimeDiff{ timeOut } < TimeDiff{ 0 }) {
                it = settingsMap.erase(it);
            } else {
                ++it;
            }
        }
    }
};

} // namespace fair::graph

#endif // GRAPH_PROTOTYPE_TRANSACTIONS_HPP
