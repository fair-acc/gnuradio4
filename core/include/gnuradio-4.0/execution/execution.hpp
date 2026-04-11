#ifndef GNURADIO_EXECUTION_HPP
#define GNURADIO_EXECUTION_HPP

#include <atomic>
#include <concepts>
#include <condition_variable>
#include <exception>
#include <memory>
#include <mutex>
#include <optional>
#include <tuple>
#include <type_traits>
#include <utility>

/**
 * @brief Minimal C++26 std::execution (P2300) sender/receiver layer for GNU Radio 4.
 *
 * Implements 8 algorithms with exact P2300 naming: just, then, let_value, when_all,
 * continues_on, bulk, sync_wait, schedule. Pipe syntax supported. Migration to
 * std::execution is a namespace swap when compiler support lands.
 *
 * Each sender declares `value_tuple_t` so sync_wait auto-deduces its return type.
 * All algorithms are lazy; sync_wait is the only blocking primitive.
 */
namespace gr::execution {

template<typename... Sigs>
struct completion_signatures {};

struct set_value_t {
    template<typename R, typename... Vs>
    void operator()(R&& r, Vs&&... vs) const noexcept {
        std::forward<R>(r).set_value(std::forward<Vs>(vs)...);
    }
};

struct set_error_t {
    template<typename R, typename E>
    void operator()(R&& r, E&& e) const noexcept {
        std::forward<R>(r).set_error(std::forward<E>(e));
    }
};

struct set_stopped_t {
    template<typename R>
    void operator()(R&& r) const noexcept {
        std::forward<R>(r).set_stopped();
    }
};

inline constexpr set_value_t   set_value{};
inline constexpr set_error_t   set_error{};
inline constexpr set_stopped_t set_stopped{};

template<typename S>
using value_tuple_of = typename std::remove_cvref_t<S>::value_tuple_t;

namespace detail {

struct NullReceiver {
    template<typename... Vs>
    void set_value(Vs&&...) noexcept {}
    void set_error(std::exception_ptr) noexcept {}
    void set_stopped() noexcept {}
};

template<typename Recv, typename... Vs>
struct ForwardReceiver {
    std::shared_ptr<std::tuple<Recv, std::tuple<Vs...>>> captured;

    void set_value() noexcept {
        auto& [recv, vals] = *captured;
        std::apply([&](auto&&... v) { gr::execution::set_value(std::move(recv), std::move(v)...); }, std::move(vals));
    }

    void set_error(std::exception_ptr e) noexcept { gr::execution::set_error(std::move(std::get<0>(*captured)), std::move(e)); }

    void set_stopped() noexcept { gr::execution::set_stopped(std::move(std::get<0>(*captured))); }
};

} // namespace detail

template<typename S>
concept sender = requires {
    typename std::remove_cvref_t<S>::completion_signatures_t;
    typename std::remove_cvref_t<S>::value_tuple_t;
};

template<typename R>
concept receiver = requires(std::remove_cvref_t<R>& r, std::exception_ptr e) {
    { r.set_stopped() } noexcept;
    { r.set_error(std::move(e)) } noexcept;
};

template<typename O>
concept operation_state = requires(O& o) {
    { o.start() } noexcept;
};

template<typename S>
concept scheduler = requires(S&& s) {
    { std::forward<S>(s).schedule() } -> sender;
};

// just(Vs...) -> sender<Vs...>

template<typename... Vs>
struct JustSender {
    using completion_signatures_t = completion_signatures<set_value_t(Vs...), set_error_t(std::exception_ptr)>;
    using value_tuple_t           = std::tuple<Vs...>;

    std::tuple<Vs...> values;

    template<receiver R>
    struct Op {
        std::tuple<Vs...>      values;
        std::remove_cvref_t<R> recv;

        void start() noexcept {
            try {
                std::apply([this](auto&&... vs) { gr::execution::set_value(std::move(recv), std::forward<decltype(vs)>(vs)...); }, std::move(values));
            } catch (...) {
                gr::execution::set_error(std::move(recv), std::current_exception());
            }
        }
    };

    template<receiver R>
    [[nodiscard]] auto connect(R&& r) const& {
        return Op<R>{values, std::forward<R>(r)};
    }

    template<receiver R>
    [[nodiscard]] auto connect(R&& r) && {
        return Op<R>{std::move(values), std::forward<R>(r)};
    }
};

struct just_t {
    template<typename... Vs>
    [[nodiscard]] auto operator()(Vs&&... vs) const {
        return JustSender<std::decay_t<Vs>...>{.values = {std::forward<Vs>(vs)...}};
    }
};

inline constexpr just_t just{};

// then(sender, f) -> sender

template<sender S, typename F>
struct ThenSender {
    using ResultT                 = decltype(std::apply(std::declval<F>(), std::declval<value_tuple_of<S>>()));
    using value_tuple_t           = std::conditional_t<std::is_void_v<ResultT>, std::tuple<>, std::tuple<ResultT>>;
    using completion_signatures_t = completion_signatures<>;

    S prev;
    F func;

    template<receiver R>
    struct Adapter {
        F                      func;
        std::remove_cvref_t<R> recv;

        template<typename... Vs>
        void set_value(Vs&&... vs) noexcept {
            try {
                if constexpr (std::is_void_v<std::invoke_result_t<F, Vs...>>) {
                    std::invoke(std::move(func), std::forward<Vs>(vs)...);
                    gr::execution::set_value(std::move(recv));
                } else {
                    gr::execution::set_value(std::move(recv), std::invoke(std::move(func), std::forward<Vs>(vs)...));
                }
            } catch (...) {
                gr::execution::set_error(std::move(recv), std::current_exception());
            }
        }

        void set_error(std::exception_ptr e) noexcept { gr::execution::set_error(std::move(recv), std::move(e)); }
        void set_stopped() noexcept { gr::execution::set_stopped(std::move(recv)); }
    };

    template<receiver R>
    struct Op {
        using InnerOp = decltype(std::declval<S>().connect(std::declval<Adapter<R>>()));
        InnerOp inner;
        void    start() noexcept { inner.start(); }
    };

    template<receiver R>
    [[nodiscard]] auto connect(R&& r) && {
        auto adapter = Adapter<R>{std::move(func), std::forward<R>(r)};
        return Op<R>{.inner = std::move(prev).connect(std::move(adapter))};
    }
};

namespace detail {
template<typename F>
struct ThenClosure {
    F func;
    template<sender S>
    [[nodiscard]] friend auto operator|(S&& s, ThenClosure&& c) {
        return ThenSender<std::decay_t<S>, F>{.prev = std::forward<S>(s), .func = std::move(c.func)};
    }
};
} // namespace detail

struct then_t {
    template<sender S, typename F>
    [[nodiscard]] auto operator()(S&& s, F&& f) const {
        return ThenSender<std::decay_t<S>, std::decay_t<F>>{.prev = std::forward<S>(s), .func = std::forward<F>(f)};
    }
    template<typename F>
    [[nodiscard]] auto operator()(F&& f) const {
        return detail::ThenClosure<std::decay_t<F>>{.func = std::forward<F>(f)};
    }
};

inline constexpr then_t then{};

// let_value(sender, f) -> sender    f returns a new sender (flatmap)

template<sender S, typename F>
struct LetValueSender {
    using value_tuple_t           = value_tuple_of<decltype(std::apply(std::declval<F>(), std::declval<value_tuple_of<S>>()))>;
    using completion_signatures_t = completion_signatures<>;

    S prev;
    F func;

    template<receiver R>
    struct Adapter {
        F                      func;
        std::remove_cvref_t<R> recv;

        template<typename... Vs>
        void set_value(Vs&&... vs) noexcept {
            try {
                auto nextOp = std::invoke(std::move(func), std::forward<Vs>(vs)...).connect(std::move(recv));
                nextOp.start();
            } catch (...) {
                gr::execution::set_error(std::move(recv), std::current_exception());
            }
        }

        void set_error(std::exception_ptr e) noexcept { gr::execution::set_error(std::move(recv), std::move(e)); }
        void set_stopped() noexcept { gr::execution::set_stopped(std::move(recv)); }
    };

    template<receiver R>
    struct Op {
        using InnerOp = decltype(std::declval<S>().connect(std::declval<Adapter<R>>()));
        InnerOp inner;
        void    start() noexcept { inner.start(); }
    };

    template<receiver R>
    [[nodiscard]] auto connect(R&& r) && {
        auto adapter = Adapter<R>{std::move(func), std::forward<R>(r)};
        return Op<R>{.inner = std::move(prev).connect(std::move(adapter))};
    }
};

namespace detail {
template<typename F>
struct LetValueClosure {
    F func;
    template<sender S>
    [[nodiscard]] friend auto operator|(S&& s, LetValueClosure&& c) {
        return LetValueSender<std::decay_t<S>, F>{.prev = std::forward<S>(s), .func = std::move(c.func)};
    }
};
} // namespace detail

struct let_value_t {
    template<sender S, typename F>
    [[nodiscard]] auto operator()(S&& s, F&& f) const {
        return LetValueSender<std::decay_t<S>, std::decay_t<F>>{.prev = std::forward<S>(s), .func = std::forward<F>(f)};
    }
    template<typename F>
    [[nodiscard]] auto operator()(F&& f) const {
        return detail::LetValueClosure<std::decay_t<F>>{.func = std::forward<F>(f)};
    }
};

inline constexpr let_value_t let_value{};

// when_all(senders...) -> sender<>    concurrent fan-out, void result

template<sender... Ss>
struct WhenAllSender {
    using completion_signatures_t = completion_signatures<set_value_t(), set_error_t(std::exception_ptr), set_stopped_t()>;
    using value_tuple_t           = std::tuple<>;

    std::tuple<Ss...> senders;

    struct SharedState {
        std::atomic<std::size_t> remaining;
        std::exception_ptr       error;
        bool                     stopped = false;
        std::mutex               mtx;

        explicit SharedState(std::size_t n) : remaining(n) {}
    };

    template<typename Recv>
    struct SubRecv {
        std::shared_ptr<SharedState> state;
        std::shared_ptr<Recv>        recv;

        template<typename... Vs>
        void set_value(Vs&&...) noexcept {
            complete();
        }

        void set_error(std::exception_ptr e) noexcept {
            {
                std::lock_guard lk(state->mtx);
                if (!state->error) {
                    state->error = std::move(e);
                }
            }
            complete();
        }

        void set_stopped() noexcept {
            {
                std::lock_guard lk(state->mtx);
                state->stopped = true;
            }
            complete();
        }

    private:
        void complete() noexcept {
            if (state->remaining.fetch_sub(1) != 1) {
                return;
            }
            if (state->error) {
                gr::execution::set_error(std::move(*recv), std::move(state->error));
            } else if (state->stopped) {
                gr::execution::set_stopped(std::move(*recv));
            } else {
                gr::execution::set_value(std::move(*recv));
            }
        }
    };

    template<receiver R>
    struct Op {
        std::shared_ptr<SharedState>            state;
        std::shared_ptr<std::remove_cvref_t<R>> recv;
        std::function<void()>                   startAll;
        void                                    start() noexcept { startAll(); }
    };

    template<receiver R>
    [[nodiscard]] auto connect(R&& r) && {
        constexpr auto N     = sizeof...(Ss);
        auto           state = std::make_shared<SharedState>(N);

        using Recv = std::remove_cvref_t<R>;
        auto recv  = std::make_shared<Recv>(std::forward<R>(r));

        auto startAll = [senders = std::move(senders), state, recv]() mutable { [&]<std::size_t... Is>(std::index_sequence<Is...>) { ([&] { std::move(std::get<Is>(senders)).connect(SubRecv<Recv>{state, recv}).start(); }(), ...); }(std::make_index_sequence<N>{}); };

        return Op<R>{.state = state, .recv = recv, .startAll = std::move(startAll)};
    }
};

struct when_all_t {
    template<sender... Ss>
    [[nodiscard]] auto operator()(Ss&&... ss) const {
        return WhenAllSender<std::decay_t<Ss>...>{.senders = {std::forward<Ss>(ss)...}};
    }
};

inline constexpr when_all_t when_all{};

// continues_on(sender, scheduler) -> sender    downstream work runs on scheduler

template<sender S, scheduler Sched>
struct ContinuesOnSender {
    using value_tuple_t           = value_tuple_of<S>;
    using completion_signatures_t = completion_signatures<>;

    S     prev;
    Sched sched;

    template<receiver R>
    struct Adapter {
        Sched                  sched;
        std::remove_cvref_t<R> recv;

        template<typename... Vs>
        void set_value(Vs&&... vs) noexcept {
            try {
                auto captured = std::make_shared<std::tuple<std::remove_cvref_t<R>, std::tuple<std::decay_t<Vs>...>>>(std::move(recv), std::tuple{std::forward<Vs>(vs)...});
                std::move(sched.schedule()).connect(detail::ForwardReceiver<std::remove_cvref_t<R>, std::decay_t<Vs>...>{captured}).start();
            } catch (...) {
                gr::execution::set_error(std::move(recv), std::current_exception());
            }
        }

        void set_error(std::exception_ptr e) noexcept { gr::execution::set_error(std::move(recv), std::move(e)); }
        void set_stopped() noexcept { gr::execution::set_stopped(std::move(recv)); }
    };

    template<receiver R>
    struct Op {
        using InnerOp = decltype(std::declval<S>().connect(std::declval<Adapter<R>>()));
        InnerOp inner;
        void    start() noexcept { inner.start(); }
    };

    template<receiver R>
    [[nodiscard]] auto connect(R&& r) && {
        auto adapter = Adapter<R>{std::move(sched), std::forward<R>(r)};
        return Op<R>{.inner = std::move(prev).connect(std::move(adapter))};
    }
};

namespace detail {
template<scheduler Sched>
struct ContinuesOnClosure {
    Sched sched;
    template<sender S>
    [[nodiscard]] friend auto operator|(S&& s, ContinuesOnClosure&& c) {
        return ContinuesOnSender<std::decay_t<S>, Sched>{.prev = std::forward<S>(s), .sched = std::move(c.sched)};
    }
};
} // namespace detail

struct continues_on_t {
    template<sender S, scheduler Sched>
    [[nodiscard]] auto operator()(S&& s, Sched&& sched) const {
        return ContinuesOnSender<std::decay_t<S>, std::decay_t<Sched>>{.prev = std::forward<S>(s), .sched = std::forward<Sched>(sched)};
    }
    template<scheduler Sched>
    [[nodiscard]] auto operator()(Sched&& sched) const {
        return detail::ContinuesOnClosure<std::decay_t<Sched>>{.sched = std::forward<Sched>(sched)};
    }
};

inline constexpr continues_on_t continues_on{};

// bulk(sender, shape, f) -> sender    data-parallel invocation

template<sender S, typename Shape, typename F>
struct BulkSender {
    using value_tuple_t           = value_tuple_of<S>;
    using completion_signatures_t = completion_signatures<>;

    S     prev;
    Shape shape;
    F     func;

    template<receiver R>
    struct Adapter {
        Shape                  shape;
        F                      func;
        std::remove_cvref_t<R> recv;

        template<typename... Vs>
        void set_value(Vs&&... vs) noexcept {
            try {
                for (Shape i = 0; i < shape; ++i) {
                    func(i, vs...);
                }
                gr::execution::set_value(std::move(recv), std::forward<Vs>(vs)...);
            } catch (...) {
                gr::execution::set_error(std::move(recv), std::current_exception());
            }
        }

        void set_error(std::exception_ptr e) noexcept { gr::execution::set_error(std::move(recv), std::move(e)); }
        void set_stopped() noexcept { gr::execution::set_stopped(std::move(recv)); }
    };

    template<receiver R>
    struct Op {
        using InnerOp = decltype(std::declval<S>().connect(std::declval<Adapter<R>>()));
        InnerOp inner;
        void    start() noexcept { inner.start(); }
    };

    template<receiver R>
    [[nodiscard]] auto connect(R&& r) && {
        auto adapter = Adapter<R>{shape, std::move(func), std::forward<R>(r)};
        return Op<R>{.inner = std::move(prev).connect(std::move(adapter))};
    }
};

namespace detail {
template<typename Shape, typename F>
struct BulkClosure {
    Shape shape;
    F     func;
    template<sender S>
    [[nodiscard]] friend auto operator|(S&& s, BulkClosure&& c) {
        return BulkSender<std::decay_t<S>, Shape, F>{.prev = std::forward<S>(s), .shape = c.shape, .func = std::move(c.func)};
    }
};
} // namespace detail

struct bulk_t {
    template<sender S, typename Shape, typename F>
    [[nodiscard]] auto operator()(S&& s, Shape shape, F&& f) const {
        return BulkSender<std::decay_t<S>, Shape, std::decay_t<F>>{.prev = std::forward<S>(s), .shape = shape, .func = std::forward<F>(f)};
    }
    template<typename Shape, typename F>
    [[nodiscard]] auto operator()(Shape shape, F&& f) const {
        return detail::BulkClosure<Shape, std::decay_t<F>>{.shape = shape, .func = std::forward<F>(f)};
    }
};

inline constexpr bulk_t bulk{};

// sync_wait(sender) -> optional<tuple<Vs...>>

namespace detail {

template<typename Tuple>
struct SyncWaitState {
    std::mutex              mtx;
    std::condition_variable cv;
    bool                    done    = false;
    bool                    stopped = false;
    std::exception_ptr      error;
    std::optional<Tuple>    value;
};

template<typename Tuple>
struct SyncWaitReceiver {
    SyncWaitState<Tuple>* state;

    void set_value() noexcept
    requires std::same_as<Tuple, std::tuple<>>
    {
        std::lock_guard lk(state->mtx);
        state->value.emplace();
        state->done = true;
        state->cv.notify_one();
    }

    template<typename... Vs>
    void set_value(Vs&&... vs) noexcept
    requires(!std::same_as<Tuple, std::tuple<>>)
    {
        std::lock_guard lk(state->mtx);
        state->value.emplace(std::forward<Vs>(vs)...);
        state->done = true;
        state->cv.notify_one();
    }

    void set_error(std::exception_ptr e) noexcept {
        std::lock_guard lk(state->mtx);
        state->error = std::move(e);
        state->done  = true;
        state->cv.notify_one();
    }

    void set_stopped() noexcept {
        std::lock_guard lk(state->mtx);
        state->stopped = true;
        state->done    = true;
        state->cv.notify_one();
    }
};

} // namespace detail

struct sync_wait_t {
    template<sender S>
    [[nodiscard]] auto operator()(S&& s) const -> std::optional<value_tuple_of<S>> {
        using Tuple = value_tuple_of<S>;
        detail::SyncWaitState<Tuple>    state;
        detail::SyncWaitReceiver<Tuple> recv{&state};
        auto                            op = std::forward<S>(s).connect(std::move(recv));
        op.start();

        std::unique_lock lk(state.mtx);
        state.cv.wait(lk, [&] { return state.done; });

        if (state.error) {
            std::rethrow_exception(state.error);
        }
        if (state.stopped) {
            return std::nullopt;
        }
        return std::move(state.value);
    }
};

inline constexpr sync_wait_t sync_wait{};

// schedule(scheduler) -> sender

struct schedule_t {
    template<scheduler Sched>
    [[nodiscard]] auto operator()(Sched&& sched) const {
        return std::forward<Sched>(sched).schedule();
    }
};

inline constexpr schedule_t schedule{};

} // namespace gr::execution

#endif // GNURADIO_EXECUTION_HPP
