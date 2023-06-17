#ifndef GNURADIO_DATA_SINK_HPP
#define GNURADIO_DATA_SINK_HPP

#include "circular_buffer.hpp"
#include "dataset.hpp"
#include "node.hpp"
#include "tag.hpp"

#include <any>
#include <chrono>

namespace fair::graph {

enum class blocking_mode {
    NonBlocking,
    Blocking
};

enum class trigger_observer_state {
    Start, ///< Start a new dataset
    Stop, ///< Finish dataset
    StopAndStart, ///< Finish pending dataset, start a new one
    Ignore ///< Ignore tag
};

template<typename T>
concept TriggerPredicate = requires(const T p, tag_t tag) {
    {p(tag)} -> std::convertible_to<bool>;
};

template<typename T>
concept TriggerObserver = requires(T o, tag_t tag) {
    {o(tag)} -> std::convertible_to<trigger_observer_state>;
};

template<typename T>
concept TriggerObserverFactory = requires(T f) {
    {f()} -> TriggerObserver;
};

template<typename T>
class data_sink;

class data_sink_registry {
    std::mutex mutex;
    std::vector<std::any> sinks;

public:
    // TODO this shouldn't be a singleton but associated with the flow graph (?)
    static data_sink_registry& instance() {
        static data_sink_registry s_instance;
        return s_instance;
    }

    template<typename T>
    void register_sink(data_sink<T> *sink) {
        std::lock_guard lg{mutex};
        sinks.push_back(sink);
    }

    template<typename T>
    void unregister_sink(data_sink<T> *sink) {
        std::lock_guard lg{mutex};
        std::erase_if(sinks, [sink](const std::any &v) {
            try {
                return std::any_cast<data_sink<T> *>(v) == sink;
            } catch (...) {
                return false;
            }
        });
    }

    template<typename T>
    std::shared_ptr<typename data_sink<T>::poller> get_streaming_poller(std::string_view name, blocking_mode block = blocking_mode::NonBlocking) {
        std::lock_guard lg{mutex};
        auto sink = find_sink<T>(name);
        return sink ? sink->get_streaming_poller(block) : nullptr;
    }

    template<typename T, TriggerPredicate P>
    std::shared_ptr<typename data_sink<T>::dataset_poller> get_trigger_poller(std::string_view name, P p, std::size_t pre_samples, std::size_t post_samples, blocking_mode block = blocking_mode::NonBlocking) {
        std::lock_guard lg{mutex};
        auto sink = find_sink<T>(name);
        return sink ? sink->get_trigger_poller(std::forward<P>(p), pre_samples, post_samples, block) : nullptr;
    }

    template<typename T, TriggerObserverFactory F>
    std::shared_ptr<typename data_sink<T>::dataset_poller> get_multiplexed_poller(std::string_view name, F triggerObserverFactory, std::size_t maximum_window_size, blocking_mode block = blocking_mode::NonBlocking) {
        std::lock_guard lg{mutex};
        auto sink = find_sink<T>(name);
        return sink ? sink->get_multiplexed_poller(std::forward<F>(triggerObserverFactory), maximum_window_size, block) : nullptr;
    }

    template<typename T, TriggerPredicate P>
    std::shared_ptr<typename data_sink<T>::dataset_poller> get_snapshot_poller(std::string_view name, P p, std::chrono::nanoseconds delay, blocking_mode block = blocking_mode::NonBlocking) {
        std::lock_guard lg{mutex};
        auto sink = find_sink<T>(name);
        return sink ? sink->get_snapshot_poller(std::forward<P>(p), delay, block) : nullptr;
    }

    template<typename T, typename Callback>
    bool register_streaming_callback(std::string_view name, std::size_t max_chunk_size, Callback callback) {
        std::lock_guard lg{mutex};
        auto sink = find_sink<T>(name);
        if (!sink) {
            return false;
        }

        sink->register_streaming_callback(max_chunk_size, std::forward<Callback>(callback));
        return true;
    }

    template<typename T, TriggerPredicate P, typename Callback>
    bool register_trigger_callback(std::string_view name, P p, std::size_t pre_samples, std::size_t post_samples, Callback callback) {
        std::lock_guard lg{mutex};
        auto sink = find_sink<T>(name);
        if (!sink) {
            return false;
        }

        sink->register_trigger_callback(std::forward<P>(p), pre_samples, post_samples, std::forward<Callback>(callback));
        return true;
    }

    template<typename T, TriggerObserver O, typename Callback>
    bool register_multiplexed_callback(std::string_view name, std::size_t maximum_window_size, Callback callback) {
        std::lock_guard lg{mutex};
        auto sink = find_sink<T>(name);
        if (!sink) {
            return false;
        }

        sink->template register_multiplexed_callback<O, Callback>(maximum_window_size, std::move(callback));
        return true;
    }

    template<typename T, TriggerPredicate P, typename Callback>
    bool register_snapshot_callback(std::string_view name, P p, std::chrono::nanoseconds delay, Callback callback) {
        std::lock_guard lg{mutex};
        auto sink = find_sink<T>(name);
        if (!sink) {
            return false;
        }

        sink->template register_snapshot_callback<P, Callback>(std::forward<P>(p), delay, std::forward<Callback>(callback));
        return true;
    }

private:
    template<typename T>
    data_sink<T>* find_sink(std::string_view name) {
        const auto it = std::find_if(sinks.begin(), sinks.end(), matcher<T>(name));
        if (it == sinks.end()) {
            return nullptr;
        }

        return std::any_cast<data_sink<T>*>(*it);
    }

    template<typename T>
    static auto matcher(std::string_view name) {
        return [name](const std::any &v) {
            try {
                return std::any_cast<data_sink<T>*>(v)->name() == name;
            } catch (...) {
                return false;
            }
        };
    }
};

template<typename T>
class data_sink : public node<data_sink<T>> {
public:
    IN<T>        in;
    std::size_t  n_samples_consumed = 0;
    float        sample_rate        = 10000;

    static constexpr std::size_t listener_buffer_size = 65536;

    template<typename Payload>
    struct poller_t {
        std::atomic<bool> finished = false;
        std::atomic<std::size_t> drop_count = 0;
        gr::circular_buffer<Payload> buffer = gr::circular_buffer<Payload>(listener_buffer_size);
        decltype(buffer.new_reader()) reader = buffer.new_reader();
        decltype(buffer.new_writer()) writer = buffer.new_writer();

        [[nodiscard]] bool process_bulk(std::invocable<std::span<Payload>> auto fnc) {
            const auto available = reader.available();
            if (available == 0) {
                return false;
            }

            const auto read_data = reader.get(available);
            fnc(read_data);
            reader.consume(available);
            return true;
        }

        [[nodiscard]] bool process_one(std::invocable<Payload> auto fnc) {
            const auto available = reader.available();
            if (available == 0) {
                return false;
            }

            const auto read_data = reader.get(1);
            fnc(read_data[0]);
            reader.consume(1);
            return true;
        }
    };

    using poller = poller_t<T>;
    using dataset_poller = poller_t<DataSet<T>>;

private:
    struct pending_window_t {
        tag_t trigger;
        DataSet<T> dataset;
        std::size_t pending_post_samples = 0;
    };

    struct abstract_listener_t {
        virtual ~abstract_listener_t() = default;
        virtual void set_sample_rate(float) {}
        virtual void process_bulk(std::span<const T> history, std::span<const T> data, int64_t reader_position, const std::vector<tag_t> &tags) = 0;
        virtual void flush() = 0;
    };

    template<typename Callback>
    struct continuous_listener_t : public abstract_listener_t {
        bool block = false;
        // callback-only
        std::size_t buffer_fill = 0;
        std::vector<T> buffer;

        // polling-only
        std::weak_ptr<poller> polling_handler = {};

        Callback callback;

        explicit continuous_listener_t(std::size_t max_chunk_size, Callback c)
            : buffer(max_chunk_size)
            , callback{std::forward<Callback>(c)}
        {}

        explicit continuous_listener_t(std::shared_ptr<poller> poller, bool do_block)
            : block(do_block)
            , polling_handler{std::move(poller)}
        {}

        void process_bulk(std::span<const T>, std::span<const T> data, int64_t /*reader_position*/, const std::vector<tag_t> &tags) override {
            if constexpr (!std::is_same_v<Callback, bool>) {
                // if there's pending data, fill buffer and send out
                if (buffer_fill > 0) {
                    const auto n = std::min(data.size(), buffer.size() - buffer_fill);
                    std::copy(data.begin(), data.begin() + n, buffer.begin() + buffer_fill);
                    buffer_fill += n;
                    if (buffer_fill == buffer.size()) {
                        callback(std::span(buffer));
                        buffer_fill = 0;
                    }

                    data = data.last(data.size() - n);
                }

                // send out complete chunks directly
                while (data.size() > buffer.size()) {
                    callback(data.first(buffer.size()));
                    data = data.last(data.size() - buffer.size());
                }

                // write remaining data to the buffer
                if (!data.empty()) {
                    std::copy(data.begin(), data.end(), buffer.begin());
                    buffer_fill = data.size();
                }
            } else {
                auto poller = polling_handler.lock();
                if (!poller) {
                    // TODO someone remove this listener from the list
                    return;
                }

                if (block) {
                    auto write_data = poller->writer.reserve_output_range(data.size());
                    std::copy(data.begin(), data.end(), write_data.begin());
                    write_data.publish(write_data.size());
                } else {
                    const auto can_write = poller->writer.available();
                    auto to_write = std::min(data.size(), can_write);
                    poller->drop_count += data.size() - can_write;
                    if (to_write > 0) {
                        auto write_data = poller->writer.reserve_output_range(to_write);
                        const auto sub = data.first(to_write);
                        std::copy(sub.begin(), sub.end(), write_data.begin());
                        write_data.publish(write_data.size());
                    }
                }
            }
        }

        void flush() override {
            if constexpr (!std::is_same_v<Callback, bool>) {
                if (buffer_fill > 0) {
                    callback(std::span(buffer).first(buffer_fill));
                    buffer_fill = 0;
                }
            } else {
                if (auto p = polling_handler.lock()) {
                    p->finished = true;
                }
            }
        }
    };

    template<typename Callback, TriggerPredicate P>
    struct trigger_listener_t : public abstract_listener_t {
        bool block = false;
        std::size_t pre_samples = 0;
        std::size_t post_samples = 0;

        P trigger_predicate = {};
        std::deque<pending_window_t> pending_trigger_windows; // triggers that still didn't receive all their data
        std::weak_ptr<dataset_poller> polling_handler = {};

        Callback callback;

        explicit trigger_listener_t(P predicate, std::shared_ptr<dataset_poller> handler, std::size_t pre, std::size_t post, bool do_block)
            : block(do_block)
            , pre_samples(pre)
            , post_samples(post)
            , trigger_predicate(std::forward<P>(predicate))
            , polling_handler{std::move(handler)}
        {}

        explicit trigger_listener_t(P predicate, std::size_t pre, std::size_t post, Callback cb)
            : pre_samples(pre)
            , post_samples(post)
            , trigger_predicate(std::forward<P>(predicate))
            , callback{std::forward<Callback>(cb)}
        {}

        // TODO all the dataset-based listeners could share publish_dataset and parts of flush (closing pollers),
        // but if we want to use different datastructures/pass additional info, this might become moot again, so
        // I leave it as is for now.
        inline void publish_dataset(DataSet<T> &&data) {
            if constexpr (!std::is_same_v<Callback, bool>) {
                callback(std::move(data));
            } else {
                auto poller = polling_handler.lock();
                if (!poller) {
                    return;
                }

                auto write_data = poller->writer.reserve_output_range(1);
                if (block) {
                    write_data[0] = std::move(data);
                    write_data.publish(1);
                } else {
                    if (poller->writer.available() > 0) {
                        write_data[0] = std::move(data);
                        write_data.publish(1);
                    } else {
                        poller->drop_count++;
                    }
                }
            }
        }

        void process_bulk(std::span<const T> history, std::span<const T> in_data, int64_t reader_position, const std::vector<tag_t> &tags) override {
            auto filtered = tags; // should use views::filter once that is working everywhere
            std::erase_if(filtered, [this](const auto &tag) {
                return !trigger_predicate(tag);
            });
            for (const auto &trigger : filtered) {
                // TODO fill dataset with metadata etc.
                DataSet<T> dataset;
                dataset.timing_events = {{trigger}};
                dataset.signal_values.reserve(pre_samples + post_samples); // TODO maybe make the circ. buffer smaller but preallocate these
                pending_trigger_windows.push_back({.trigger = trigger, .dataset = std::move(dataset), .pending_post_samples = post_samples});
            }

            auto window = pending_trigger_windows.begin();
            while (window != pending_trigger_windows.end()) {
                auto &dataset = window->dataset;
                const auto window_offset = window->trigger.index - reader_position;

                if (window_offset >= 0 && dataset.signal_values.empty()) { // new trigger, write history
                    // old history: pre-trigger data from previous in_data (if available)
                    const auto old_history_size = std::max(static_cast<std::int64_t>(pre_samples) - window_offset, std::int64_t{0});
                    const auto available = std::min(static_cast<std::size_t>(old_history_size), history.size());
                    const auto old_history_view = history.last(available);
                    dataset.signal_values.insert(dataset.signal_values.end(), old_history_view.begin(), old_history_view.end());

                    // new history: pre-trigger samples from the current in_data
                    const auto new_history_size = pre_samples - old_history_size;
                    const auto new_history_view = in_data.subspan(window_offset - new_history_size, new_history_size);
                    dataset.signal_values.insert(dataset.signal_values.end(), new_history_view.begin(), new_history_view.end());
                }

                // write missing post-samples
                const auto previous_post_samples = post_samples - window->pending_post_samples;
                const auto first_requested = window_offset + previous_post_samples;
                const auto last_requested = window_offset + post_samples - 1;
                const auto last_available = std::min(last_requested, in_data.size() - 1);
                const auto post_sample_view = in_data.subspan(first_requested, last_available - first_requested + 1);
                dataset.signal_values.insert(dataset.signal_values.end(), post_sample_view.begin(), post_sample_view.end());
                window->pending_post_samples -= post_sample_view.size();

                if (window->pending_post_samples == 0) {
                    publish_dataset(std::move(dataset));
                    window = pending_trigger_windows.erase(window);
                } else {
                    ++window;
                }
            }
        }

        void flush() override {
            for (auto &window : pending_trigger_windows) {
                if (!window.dataset.signal_values.empty()) {
                    publish_dataset(std::move(window.dataset));
                }
            }
            pending_trigger_windows.clear();
            if (auto p = polling_handler.lock()) {
                p->finished = true;
            }
        }
    };

    template<typename Callback, TriggerObserverFactory F>
    struct multiplexed_listener_t : public abstract_listener_t {
        bool block = false;
        F observerFactory;
        decltype(observerFactory()) observer;
        std::optional<DataSet<T>> pending_dataset;
        std::size_t maximum_window_size;
        std::weak_ptr<dataset_poller> polling_handler = {};
        Callback callback;

        explicit multiplexed_listener_t(F factory, std::size_t max_window_size, Callback cb) : observerFactory(factory), observer(observerFactory()), maximum_window_size(max_window_size), callback(cb) {}
        explicit multiplexed_listener_t(F factory, std::size_t max_window_size, std::shared_ptr<dataset_poller> handler, bool do_block) : observerFactory(factory), observer(observerFactory()), maximum_window_size(max_window_size), polling_handler{std::move(handler)}, block(do_block) {}

        inline void publish_dataset(DataSet<T> &&data) {
            if constexpr (!std::is_same_v<Callback, bool>) {
                callback(std::move(data));
            } else {
                auto poller = polling_handler.lock();
                if (!poller) {
                    return;
                }

                auto write_data = poller->writer.reserve_output_range(1);
                if (block) {
                    write_data[0] = std::move(data);
                    write_data.publish(1);
                } else {
                    if (poller->writer.available() > 0) {
                        write_data[0] = std::move(data);
                        write_data.publish(1);
                    } else {
                        poller->drop_count++;
                    }
                }
            }
        }

        inline void fill_pending_dataset(std::span<const T> in_data, int64_t reader_position, int64_t last_sample) {
            const auto max_samples = static_cast<int64_t>(maximum_window_size - pending_dataset->signal_values.size());
            const auto first_sample = std::max(pending_dataset->timing_events[0][0].index - reader_position, int64_t{0});
            const auto actual_last_sample = std::min(first_sample + max_samples - 1, last_sample);
            if (actual_last_sample >= first_sample) {
                pending_dataset->signal_values.insert(pending_dataset->signal_values.end(), in_data.begin() + first_sample, in_data.begin() + actual_last_sample + 1);
            }
        }

        void process_bulk(std::span<const T>, std::span<const T> in_data, int64_t reader_position, const std::vector<tag_t> &tags) override {
            for (const auto &tag :tags) {
                const auto obsr = observer(tag);
                // TODO set proper error state instead of throwing
                if (obsr == trigger_observer_state::Stop || obsr == trigger_observer_state::StopAndStart) {
                    if (obsr == trigger_observer_state::Stop && !pending_dataset) {
                        throw std::runtime_error("multiplexed: Stop without start");
                    }

                    pending_dataset->timing_events[0].push_back(tag);
                    fill_pending_dataset(in_data, reader_position, tag.index - reader_position - 1);
                    publish_dataset(std::move(*pending_dataset));
                    pending_dataset.reset();
                }
                if (obsr == trigger_observer_state::Start || obsr == trigger_observer_state::StopAndStart) {
                    if (obsr == trigger_observer_state::Start && pending_dataset) {
                        throw std::runtime_error("multiplexed: Two starts without stop");
                    }
                    pending_dataset = DataSet<T>();
                    pending_dataset->signal_values.reserve(maximum_window_size); // TODO might be too much?
                    pending_dataset->timing_events = {{tag}};
                }
            }
            if (pending_dataset) {
                fill_pending_dataset(in_data, reader_position, in_data.size() - 1);
                if (pending_dataset->signal_values.size() == maximum_window_size) {
                    publish_dataset(std::move(*pending_dataset));
                    pending_dataset.reset();
                }
            }
        }

        void flush() override {
            if (pending_dataset) {
                    publish_dataset(std::move(*pending_dataset));
                    pending_dataset.reset();
            }
            if (auto p = polling_handler.lock()) {
                p->finished = true;
            }
        }
    };

    struct pending_snapshot {
        tag_t tag;
        tag_t::index_type requested_sample;
    };

    template<typename Callback, TriggerPredicate P>
    struct snapshot_listener_t : public abstract_listener_t {
        bool block = false;
        std::chrono::nanoseconds time_delay;
        tag_t::index_type sample_delay = 0;
        P trigger_predicate = {};
        std::deque<pending_snapshot> pending;
        std::weak_ptr<dataset_poller> polling_handler = {};
        Callback callback;

        explicit snapshot_listener_t(P p, std::chrono::nanoseconds delay, std::shared_ptr<dataset_poller> poller, bool do_block) : block(do_block), time_delay(delay), trigger_predicate(std::forward<P>(p)), polling_handler{std::move(poller)} {}
        explicit snapshot_listener_t(P p, std::chrono::nanoseconds delay, Callback cb) : trigger_predicate(std::forward<P>(p)), time_delay(std::forward<Callback>(cb)) {}

        inline void publish_dataset(DataSet<T> &&data) {
            if constexpr (!std::is_same_v<Callback, bool>) {
                callback(std::move(data));
            } else {
                auto poller = polling_handler.lock();
                if (!poller) {
                    return;
                }

                auto write_data = poller->writer.reserve_output_range(1);
                if (block) {
                    write_data[0] = std::move(data);
                    write_data.publish(1);
                } else {
                    if (poller->writer.available() > 0) {
                        write_data[0] = std::move(data);
                        write_data.publish(1);
                    } else {
                        poller->drop_count++;
                    }
                }
            }
        }

        void set_sample_rate(float r) override {
            sample_delay = std::round(std::chrono::duration_cast<std::chrono::duration<float>>(time_delay).count() * r);
        }

        void process_bulk(std::span<const T>, std::span<const T> in_data, int64_t reader_position, const std::vector<tag_t> &tags) override {
            auto triggers = tags; // should use views::filter once that is working everywhere
            std::erase_if(triggers, [this](const auto &tag) {
                return !trigger_predicate(tag);
            });

            if (!triggers.empty()) {
                for (const auto &trigger : triggers) {
                    pending.push_back({trigger, trigger.index + sample_delay});
                }
                // can be unsorted if sample_delay changed. Alternative: iterate the whole list below
                std::stable_sort(pending.begin(), pending.end(), [](const auto &lhs, const auto &rhs) { return lhs.requested_sample < rhs.requested_sample; });
            }

            auto it = pending.begin();
            while (it != pending.end()) {
                const auto rel_pos = it->requested_sample - reader_position;
                assert(rel_pos >= 0);
                if (rel_pos >= in_data.size()) {
                    break;
                }

                DataSet<T> dataset;
                dataset.timing_events = {{it->tag}};
                dataset.signal_values = {in_data[rel_pos]};
                publish_dataset(std::move(dataset));

                it = pending.erase(it);
            }
        }

        void flush() override {
            pending.clear();
            if (auto p = polling_handler.lock()) {
                p->finished = true;
            }
        }
    };

    std::deque<std::unique_ptr<abstract_listener_t>> listeners;
    std::mutex listener_mutex;

public:
    data_sink() {
        data_sink_registry::instance().register_sink(this);
    }

    ~data_sink() {
        data_sink_registry::instance().unregister_sink(this);
    }

    std::shared_ptr<poller> get_streaming_poller(blocking_mode block_mode = blocking_mode::NonBlocking) {
        std::lock_guard lg(listener_mutex);
        const auto block = block_mode == blocking_mode::Blocking;
        auto handler = std::make_shared<poller>();
        add_listener(std::make_unique<continuous_listener_t<bool>>(handler, block), block);
        return handler;
    }

    template<typename TriggerPredicate>
    std::shared_ptr<dataset_poller> get_trigger_poller(TriggerPredicate p, std::size_t pre_samples, std::size_t post_samples, blocking_mode block_mode = blocking_mode::NonBlocking) {
        const auto block = block_mode == blocking_mode::Blocking;
        auto handler = std::make_shared<dataset_poller>();
        std::lock_guard lg(listener_mutex);
        add_listener(std::make_unique<trigger_listener_t<bool, TriggerPredicate>>(std::forward<TriggerPredicate>(p), handler, pre_samples, post_samples, block), block);
        history.resize(std::max(pre_samples, history.size()));
        return handler;
    }

    template<TriggerObserverFactory F>
    std::shared_ptr<dataset_poller> get_multiplexed_poller(F triggerObserverFactory, std::size_t maximum_window_size, blocking_mode block_mode = blocking_mode::NonBlocking) {
        std::lock_guard lg(listener_mutex);
        const auto block = block_mode == blocking_mode::Blocking;
        auto handler = std::make_shared<dataset_poller>();
        add_listener(std::make_unique<multiplexed_listener_t<bool, F>>(std::move(triggerObserverFactory), maximum_window_size, handler, block), block);
        return handler;
    }

    template<TriggerPredicate P>
    std::shared_ptr<dataset_poller> get_snapshot_poller(P p, std::chrono::nanoseconds delay, blocking_mode block_mode = blocking_mode::NonBlocking) {
        const auto block = block_mode == blocking_mode::Blocking;
        auto handler = std::make_shared<dataset_poller>();
        std::lock_guard lg(listener_mutex);
        add_listener(std::make_unique<snapshot_listener_t<bool, P>>(std::forward<P>(p), delay, handler, block), block);
        return handler;
    }

    template<typename Callback>
    void register_streaming_callback(std::size_t max_chunk_size, Callback callback) {
        add_listener(std::make_unique<continuous_listener_t<Callback>>(max_chunk_size, std::forward<Callback>(callback)), false);
    }

    template<TriggerPredicate P, typename Callback>
    void register_trigger_callback(P p, std::size_t pre_samples, std::size_t post_samples, Callback callback) {
        add_listener(std::make_unique<trigger_listener_t<Callback, P>>(std::forward<P>(p), pre_samples, post_samples, std::forward<Callback>(callback)), false);
        history.resize(std::max(pre_samples, history.size()));
    }

    template<TriggerObserverFactory F, typename Callback>
    void register_multiplexed_callback(F triggerObserverFactory, std::size_t maximum_window_size, Callback callback) {
        std::lock_guard lg(listener_mutex);
        add_listener(std::make_unique<multiplexed_listener_t>(std::move(triggerObserverFactory), maximum_window_size, std::forward<Callback>(callback)), false);
    }

    template<TriggerPredicate P, typename Callback>
    void register_snapshot_callback(P p, std::chrono::nanoseconds delay, Callback callback) {
        std::lock_guard lg(listener_mutex);
        add_listener(std::make_unique<snapshot_listener_t>(std::forward<P>(p), delay, std::forward<Callback>(callback)), false);
    }

    // TODO this code should be called at the end of graph processing
    void stop() {
        std::lock_guard lg(listener_mutex);
        for (auto &listener : listeners) {
            listener->flush();
        }
    }

    [[nodiscard]] work_return_t work() {
        auto &in_port = input_port<"in">(this);
        auto &reader = in_port.streamReader();

        const auto n_readable = std::min(reader.available(), in_port.max_buffer_size());
        if (n_readable == 0) {
            return fair::graph::work_return_t::INSUFFICIENT_INPUT_ITEMS;
        }

        const auto noutput_items = std::min(listener_buffer_size, n_readable);
        const auto reader_position = reader.position() + 1;
        const auto in_data = reader.get(noutput_items);
        const auto history_view = std::span(history.begin(), history_available);
        // TODO I'm not sure why the +1 in "reader.position() + 1". Bug or do I misunderstand?
        assert(reader_position == n_samples_consumed);

        auto &tag_reader = in_port.tagReader();
        const auto n_tags = tag_reader.available();
        const auto tag_data = tag_reader.get(n_tags);
        std::vector<tag_t> tags(tag_data.begin(), tag_data.end());
        auto out_of_range = [end_pos = reader_position + noutput_items](const auto &tag) {
            return tag.index > static_cast<tag_t::index_type>(end_pos);
        };
        std::erase_if(tags, out_of_range);
        tag_reader.consume(tags.size());

        {
            std::lock_guard lg(listener_mutex);
            for (auto &listener : listeners) {
                listener->process_bulk(history, in_data, reader_position, tags);
            }

            // store potential pre-samples for triggers at the beginning of the next chunk
            // TODO should use built-in history functionality that doesn't copy (but is resizable as listeners are added)
            history_available = std::min(history.size(), noutput_items);
            const auto history_data = in_data.last(history_available);
            history.assign(history_data.begin(), history_data.end());
        }

        n_samples_consumed += noutput_items;

        if (!reader.consume(noutput_items)) {
            return work_return_t::ERROR;
        }

        return work_return_t::OK;
    }

private:
    std::vector<T> history;
    std::size_t history_available = 0;

    void add_listener(std::unique_ptr<abstract_listener_t>&& l, bool block) {
        l->set_sample_rate(sample_rate); // TODO also call when sample_rate changes
        if (block) {
            listeners.push_back(std::move(l));
        } else {
            listeners.push_front(std::move(l));
        }
    }
};

}

ENABLE_REFLECTION_FOR_TEMPLATE_FULL((typename T), (fair::graph::data_sink<T>), in, n_samples_consumed, sample_rate);

#endif
