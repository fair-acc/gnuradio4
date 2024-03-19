#ifndef GNURADIO_LIFECYCLE_HPP
#define GNURADIO_LIFECYCLE_HPP

#include <gnuradio-4.0/Message.hpp>
#include <gnuradio-4.0/meta/formatter.hpp>
#include <gnuradio-4.0/meta/utils.hpp>
#include <gnuradio-4.0/reflection.hpp>

#include <atomic>
#include <expected>
#include <source_location>
#include <string>

namespace gr ::lifecycle {
/**
 * @enum lifecycle::State enumerates the possible states of a `Scheduler` lifecycle.
 *
 * Transition between the following states is triggered by specific actions or events:
 * - `IDLE`: The initial state before the scheduler has been initialized.
 * - `INITIALISED`: The scheduler has been initialized and is ready to start running.
 * - `RUNNING`: The scheduler is actively running.
 * - `REQUESTED_PAUSE`: A pause has been requested, and the scheduler is in the process of pausing.
 * - `PAUSED`: The scheduler is paused and can be resumed or stopped.
 * - `REQUESTED_STOP`: A stop has been requested, and the scheduler is in the process of stopping.
 * - `STOPPED`: The scheduler has been stopped and can be reset or re-initialized.
 * - `ERROR`: An error state that can be reached from any state at any time, requiring a reset.
 *
 * @note All `Block<T>`-derived classes can optionally implement any subset of the lifecycle methods
 * (`start()`, `stop()`, `reset()`, `pause()`, `resume()`) to handle state changes of the `Scheduler`.
 *
 * State diagram:
 *
 *                 Block<T>()              can be reached from
 *                    │                   anywhere and anytime.
 *              ┌─────┴────┐                   ┌────┴────┐
 * ┌────────────┤   IDLE   │                   │  ERROR  │
 * │            └────┬─────┘                   └────┬────┘
 * │                 │ init()                       │ reset()
 * │                 v                              │
 * │         ┌───────┴───────┐                      │
 * ├<────────┤  INITIALISED  ├<─────────────────────┤
 * │         └───────┬───────┘                      │
 * │                 │ start()                      │
 * │                 v                              │
 * │   stop() ┌──────┴──────┐                       │  ╓
 * │ ┌────────┤   RUNNING   ├<──────────┐           │  ║
 * │ │        └─────┬───────┘           │           │  ║
 * │ │              │ pause()           │           │  ║  isActive(lifecycle::State) ─> true
 * │ │              v                   │ resume()  │  ║
 * │ │    ┌─────────┴─────────┐   ┌─────┴─────┐     │  ║
 * │ │    │  REQUESTED_PAUSE  ├──>┤  PAUSED   │     │  ║
 * │ │    └──────────┬────────┘   └─────┬─────┘     │  ╙
 * │ │               │ stop()           │ stop()    │
 * │ │               v                  │           │
 * │ │     ┌─────────┴────────┐         │           │  ╓
 * │ └────>┤  REQUESTED_STOP  ├<────────┘           │  ║
 * │       └────────┬─────────┘                     │  ║
 * │                │                               │  ║  isShuttingDown(lifecycle::State) ─> true
 * │                v                               │  ║
 * │          ┌─────┴─────┐ reset()                 │  ║
 * └─────────>│  STOPPED  ├─────────────────────────┘  ║
 *            └─────┬─────┘                            ╙
 *                  │
 *                  v
 *              ~Block<T>()
 */
enum class State : char { IDLE, INITIALISED, RUNNING, REQUESTED_PAUSE, PAUSED, REQUESTED_STOP, STOPPED, ERROR };
using enum State;

inline constexpr bool
isActive(lifecycle::State state) noexcept {
    return state == RUNNING || state == REQUESTED_PAUSE || state == PAUSED;
}

inline constexpr bool
isShuttingDown(lifecycle::State state) noexcept {
    return state == REQUESTED_STOP || state == STOPPED;
}

constexpr bool
isValidTransition(const State from, const State to) noexcept {
    if (to == State::ERROR || from == to) {
        // can transit to ERROR from any state
        return true;
    }
    switch (from) {
    case State::IDLE: return to == State::INITIALISED || to == State::REQUESTED_STOP || to == State::STOPPED;
    case State::INITIALISED: return to == State::RUNNING || to == State::REQUESTED_STOP || to == State::STOPPED;
    case State::RUNNING: return to == State::REQUESTED_PAUSE || to == State::REQUESTED_STOP;
    case State::REQUESTED_PAUSE: return to == State::PAUSED;
    case State::PAUSED: return to == State::RUNNING || to == State::REQUESTED_STOP;
    case State::REQUESTED_STOP: return to == State::STOPPED;
    case State::STOPPED: return to == State::INITIALISED;
    case State::ERROR: return to == State::INITIALISED;
    default: return false;
    }
}

enum class StorageType { ATOMIC, NON_ATOMIC };

/**
 * @brief StateMachine class template that manages the lifecycle states of a Scheduler or Block.
 * It is designed to be inherited by blocks (TDerived) to safely and effectively manage their lifecycle state transitions.
 *
 * If implemented in TDerived, the following specific lifecycle methods are called:
 * - `init()`   when transitioning from IDLE to INITIALISED
 * - `start()`  when transitioning from INITIALISED to RUNNING
 * - `stop()`   when transitioning from any `isActive(State)` to REQUESTED_STOP
 * - `pause()`  when transitioning from RUNNING to REQUESTED_PAUSE
 * - `resume()` when transitioning from PAUSED to RUNNING
 * - `reset()`  when transitioning from any state (typically ERROR or STOPPED) to INITIALISED.
 * If any of these methods throw an exception, the StateMachine transitions to the ERROR state, captures,
 * and forward the exception details.
 *
 * To react to state changes, TDerived can implement the `stateChanged(State newState)` method.
 *
 * @tparam TDerived The derived class type implementing specific lifecycle methods.
 * @tparam storageType Specifies the storage type for the state, allowing for atomic operations
 *         for thread-safe state changes. Defaults to ATOMIC.
 */
template<typename TDerived, StorageType storageType = StorageType::ATOMIC>
class StateMachine {
protected:
    using StateStorage  = std::conditional_t<storageType == StorageType::ATOMIC, std::atomic<State>, State>;
    StateStorage _state = lifecycle::State::IDLE;

    void
    setAndNotifyState(State newState) {
        if constexpr (requires(TDerived d) { d.stateChanged(newState); }) {
            static_cast<TDerived *>(this)->stateChanged(newState);
        }
        if constexpr (storageType == StorageType::ATOMIC) {
            _state.store(newState, std::memory_order_release);
            _state.notify_all();
        } else {
            _state = newState;
        }
    }

    std::string
    getBlockName() {
        if constexpr (requires(TDerived d) { d.uniqueName(); }) {
            return std::string{ static_cast<TDerived *>(this)->uniqueName() };
        } else if constexpr (requires(TDerived d) { d.unique_name; }) {
            return std::string{ static_cast<TDerived *>(this)->unique_name };
        } else {
            return "unknown block/item";
        }
    }

    template<typename TMethod>
    std::expected<void, Error>
    invokeLifecycleMethod(TMethod method, const std::source_location &location) {
        try {
            (static_cast<TDerived *>(this)->*method)();
            return {};
        } catch (const std::exception &e) {
            setAndNotifyState(State::ERROR);
            return std::unexpected(Error{ fmt::format("Block '{}' throws: {}", getBlockName(), e.what()), location });
        } catch (...) {
            setAndNotifyState(State::ERROR);
            return std::unexpected(Error{ fmt::format("Block '{}' throws: {}", getBlockName(), "unknown unnamed error"), location });
        }
    }

public:
    StateMachine() noexcept = default;

    StateMachine(StateMachine &&other) noexcept
        requires(storageType == StorageType::ATOMIC)
        : _state(other._state.load()) {} // atomic, not moving

    StateMachine(StateMachine &&other) noexcept
        requires(storageType != StorageType::ATOMIC)
        : _state(other._state) {} // plain enum

    [[nodiscard]] std::expected<void, Error>
    changeStateTo(State newState, const std::source_location location = std::source_location::current()) {
        State oldState = _state;
        if (oldState == newState || (oldState == STOPPED && newState == REQUESTED_STOP) || (oldState == PAUSED && newState == REQUESTED_PAUSE)) {
            return {};
        }

        if (!isValidTransition(oldState, newState)) {
            return std::unexpected(Error{ fmt::format("Block '{}' invalid state transition in {} from {} -> to {}", //
                                                                   getBlockName(), gr::meta::type_name<TDerived>(),              //
                                                                   magic_enum::enum_name(state()), magic_enum::enum_name(newState)),
                                                       location });
            ;
        }

        setAndNotifyState(newState);

        if constexpr (std::is_same_v<TDerived, void>) {
            return {};
        } else {
            // Call specific methods in TDerived based on the state
            if constexpr (requires(TDerived &d) { d.init(); }) {
                if (oldState == State::IDLE && newState == State::INITIALISED) {
                    return invokeLifecycleMethod(&TDerived::init, location);
                }
            }
            if constexpr (requires(TDerived &d) { d.start(); }) {
                if (oldState == State::INITIALISED && newState == State::RUNNING) {
                    return invokeLifecycleMethod(&TDerived::start, location);
                }
            }
            if constexpr (requires(TDerived &d) { d.stop(); }) {
                if (newState == State::REQUESTED_STOP) {
                    return invokeLifecycleMethod(&TDerived::stop, location);
                }
            }
            if constexpr (requires(TDerived &d) { d.pause(); }) {
                if (newState == State::REQUESTED_PAUSE) {
                    return invokeLifecycleMethod(&TDerived::pause, location);
                }
            }
            if constexpr (requires(TDerived &d) { d.resume(); }) {
                if ((oldState == State::REQUESTED_PAUSE || oldState == State::PAUSED) && newState == State::RUNNING) {
                    return invokeLifecycleMethod(&TDerived::resume, location);
                }
            }
            if constexpr (requires(TDerived &d) { d.reset(); }) {
                if (oldState != State::IDLE && newState == State::INITIALISED) {
                    return invokeLifecycleMethod(&TDerived::reset, location);
                }
            }

            return {};
        }
    }

    [[nodiscard]] State
    state() const noexcept {
        if constexpr (storageType == StorageType::ATOMIC) {
            return _state.load();
        } else {
            return _state;
        }
    }

    void
    waitOnState(State oldState)
        requires(storageType == StorageType::ATOMIC)
    {
        _state.wait(oldState);
    }
};

} // namespace gr::lifecycle

#endif // GNURADIO_LIFECYCLE_HPP
