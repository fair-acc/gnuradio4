#ifndef GNURADIO_DATA_SINK_HPP
#define GNURADIO_DATA_SINK_HPP

#include "circular_buffer.hpp"
#include "dataset.hpp"
#include "node.hpp"
#include "tag.hpp"

#include <any>

namespace fair::graph {

enum class acquisition_mode {
    Continuous,
    Triggered,
    PostMortem
};

enum class blocking_mode {
    NonBlocking,
    Blocking
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

    template<typename T, typename TriggerPredicate>
    std::shared_ptr<typename data_sink<T>::dataset_poller> get_trigger_poller(std::string_view name, TriggerPredicate p, std::size_t pre_samples, std::size_t post_samples, blocking_mode block = blocking_mode::NonBlocking) {
        std::lock_guard lg{mutex};
        auto sink = find_sink<T>(name);
        return sink ? sink->get_trigger_poller(std::move(p), pre_samples, post_samples, block) : nullptr;
    }

    template<typename T, typename Callback>
    bool register_streaming_callback(std::string_view name, std::size_t max_chunk_size, Callback callback) {
        std::lock_guard lg{mutex};
        auto sink = find_sink<T>(name);
        if (!sink) {
            return false;
        }

        sink->register_streaming_callback(max_chunk_size, std::move(callback));
        return true;
    }

    template<typename T, typename TriggerPredicate, typename Callback>
    bool register_trigger_callback(std::string_view name, TriggerPredicate p, std::size_t pre_samples, std::size_t post_samples, Callback callback) {
        std::lock_guard lg{mutex};
        auto sink = find_sink<T>(name);
        if (!sink) {
            return false;
        }

        sink->register_trigger_callback(std::move(p), pre_samples, post_samples, std::move(callback));
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
    std::size_t  n_samples_max      = -1;
    int64_t      last_tag_position  = -1;
    float        sample_rate        = -1.0f;

    static constexpr std::size_t listener_buffer_size = 65536;

    template<typename Payload>
    struct poller_t {
        std::atomic<bool> finished = false;
        std::atomic<std::size_t> drop_count = 0;
        gr::circular_buffer<Payload> buffer = gr::circular_buffer<Payload>(listener_buffer_size);
        decltype(buffer.new_reader()) reader = buffer.new_reader();
        decltype(buffer.new_writer()) writer = buffer.new_writer();

        template<typename Handler>
        [[nodiscard]] bool process_bulk(Handler fnc) {
            const auto available = reader.available();
            if (available == 0) {
                return false;
            }

            const auto read_data = reader.get(available);
            fnc(read_data);
            reader.consume(available);
            return true;
        }

        template<typename Handler>
        [[nodiscard]] bool process_one(Handler fnc) {
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

    template<typename Callback, typename TriggerPredicate>
    struct trigger_listener_t : public abstract_listener_t {
        bool block = false;
        std::size_t pre_samples = 0;
        std::size_t post_samples = 0;

        TriggerPredicate trigger_predicate = {};
        std::deque<pending_window_t> pending_trigger_windows; // triggers that still didn't receive all their data
        std::weak_ptr<dataset_poller> polling_handler = {};

        Callback callback;

        explicit trigger_listener_t(TriggerPredicate predicate, std::shared_ptr<dataset_poller> handler, std::size_t pre, std::size_t post, bool do_block)
            : block(do_block)
            , pre_samples(pre)
            , post_samples(post)
            , trigger_predicate(std::forward<TriggerPredicate>(predicate))
            , polling_handler{std::move(handler)}
        {}

        explicit trigger_listener_t(TriggerPredicate predicate, std::size_t pre, std::size_t post, Callback cb)
            : pre_samples(pre)
            , post_samples(post)
            , trigger_predicate(std::forward<TriggerPredicate>(predicate))
            , callback{std::forward<Callback>(cb)}
        {}

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

    template<typename TriggerPredicate, typename Callback>
    void register_trigger_callback(TriggerPredicate p, std::size_t pre_samples, std::size_t post_samples, Callback callback) {
        std::lock_guard lg(listener_mutex);
        add_listener(std::make_unique<trigger_listener_t<Callback, TriggerPredicate>>(std::forward<TriggerPredicate>(p), pre_samples, post_samples, std::forward<Callback>(callback)), false);
        history.resize(std::max(pre_samples, history.size()));
    }

    template<typename Callback>
    void register_streaming_callback(std::size_t max_chunk_size, Callback callback) {
        add_listener(std::make_unique<continuous_listener_t<Callback>>(max_chunk_size, std::forward<Callback>(callback)), false);
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
        if (block) {
            listeners.push_back(std::move(l));
        } else {
            listeners.push_front(std::move(l));
        }
    }
};

}

ENABLE_REFLECTION_FOR_TEMPLATE_FULL((typename T), (fair::graph::data_sink<T>), in, n_samples_consumed, n_samples_max, last_tag_position, sample_rate);

#endif
