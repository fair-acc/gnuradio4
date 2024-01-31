#ifndef GNURADIO_LIFECYCLE_HPP
#define GNURADIO_LIFECYCLE_HPP

#include <atomic>

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
 *               Block<T>()              can be reached from
 *                  │                   anywhere and anytime.
 *            ┌─────┴────┐                   ┌────┴────┐
 *            │   IDLE   │                   │  ERROR  │
 *            └────┬─────┘                   └────┬────┘
 *                 │ init()                       │ reset()
 *                 v                              │
 *         ┌───────┴───────┐                      │
 *         │  INITIALISED  ├<─────────────────────┤
 *         └───────┬───────┘                      │
 *                 │ start()                      │
 *                 v                              │
 *   stop() ┌──────┴──────┐                       │  ╓
 *   ┌──────┤   RUNNING   ├<──────────┐           │  ║
 *   │      └─────┬───────┘           │           │  ║
 *   │            │ pause()           │           │  ║  isActive(lifecycle::State) ─> true
 *   │            v                   │ resume()  │  ║
 *   │  ┌─────────┴─────────┐   ┌─────┴─────┐     │  ║
 *   │  │  REQUESTED_PAUSE  ├──>┤  PAUSED   │     │  ║
 *   │  └──────────┬────────┘   └─────┬─────┘     │  ╙
 *   │             │ stop()           │ stop()    │
 *   │             v                  │           │
 *   │   ┌─────────┴────────┐         │           │  ╓
 *   └──>┤  REQUESTED_STOP  ├<────────┘           │  ║
 *       └────────┬─────────┘                     │  ║
 *                │                               │  ║  isShuttingDown(lifecycle::State) ─> true
 *                v                               │  ║
 *          ┌─────┴─────┐ reset()                 │  ║
 *          │  STOPPED  ├─────────────────────────┘  ║
 *          └─────┬─────┘                            ╙
 *                │
 *                v
 *            ~Block<T>()
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
    if (to == State::ERROR) {
        // can transit to ERROR from any state
        return true;
    }
    switch (from) {
    case State::IDLE: return to == State::INITIALISED;
    case State::INITIALISED: return to == State::RUNNING;
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

template<typename TDerived, StorageType storageType = StorageType::ATOMIC>
class StateMachine {
protected:
    using StateStorage  = std::conditional_t<storageType == StorageType::ATOMIC, std::atomic<State>, State>;
    StateStorage _state = lifecycle::State::IDLE;

public:
    [[nodiscard]] constexpr bool
    transitionTo(State newState) {
        if (!isValidTransition(_state, newState)) {
            return false;
        }

        State oldState = _state;

        if constexpr (storageType == StorageType::ATOMIC) {
            _state.store(newState);
        } else {
            _state = newState;
        }

        if constexpr (storageType == StorageType::ATOMIC) {
            _state.notify_all();
        }

        // Call specific methods in TDerived based on the state
        if constexpr (requires(TDerived &d) { d.start(); }) {
            if (oldState == State::INITIALISED && newState == State::RUNNING) {
                static_cast<TDerived *>(this)->start();
            }
        }
        if constexpr (requires(TDerived &d) { d.stop(); }) {
            if (newState == State::REQUESTED_STOP) {
                static_cast<TDerived *>(this)->stop();
            }
        }
        if constexpr (requires(TDerived &d) { d.pause(); }) {
            if (newState == State::REQUESTED_PAUSE) {
                static_cast<TDerived *>(this)->pause();
            }
        }
        if constexpr (requires(TDerived &d) { d.resume(); }) {
            if (oldState == State::PAUSED && newState == State::RUNNING) {
                static_cast<TDerived *>(this)->resume();
            }
        }
        if constexpr (requires(TDerived &d) { d.reset(); }) {
            if (oldState == State::STOPPED && newState == State::INITIALISED) {
                static_cast<TDerived *>(this)->reset();
            }
        }

        return true;
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
