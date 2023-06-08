#ifndef GNURADIO_DATA_SINK_HPP
#define GNURADIO_DATA_SINK_HPP

#include "circular_buffer.hpp"
#include "dataset.hpp"
#include "node.hpp"
#include "tag.hpp"

#include <any>
#include <ranges>

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
    bool register_streaming_callback(std::string_view name, Callback callback) {
        std::lock_guard lg{mutex};
        auto sink = find_sink<T>(name);
        if (!sink) {
            return false;
        }

        sink->register_streaming_callback(std::move(callback));
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

    struct poller {
        std::atomic<bool> finished = false;
        std::atomic<std::size_t> drop_count = 0;
        gr::circular_buffer<T> buffer = gr::circular_buffer<T>(listener_buffer_size);
        decltype(buffer.new_reader()) reader = buffer.new_reader();

        template<typename Handler>
        [[nodiscard]] bool process(Handler fnc) {
            const auto available = reader.available();
            if (available == 0) {
                return false;
            }

            const auto read_data = reader.get(available);
            fnc(read_data);
            reader.consume(available);
            return true;
        }
    };

    struct dataset_poller {
        std::atomic<bool> finished = false;
        std::atomic<std::size_t> drop_count = 0;
        gr::circular_buffer<DataSet<T>> buffer = gr::circular_buffer<DataSet<T>>(listener_buffer_size); // TODO use other size?
        decltype(buffer.new_reader()) reader = buffer.new_reader();
        decltype(buffer.new_writer()) writer = buffer.new_writer();

        template<typename Handler>
        [[nodiscard]] bool process(Handler fnc) {
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

private:
    struct pending_window_t {
        tag_t trigger;
        DataSet<T> dataset;
        std::size_t pending_post_samples = 0;
    };

    struct listener_t {
        acquisition_mode mode = acquisition_mode::Triggered;
        std::size_t pre_samples = 0;
        std::size_t post_samples = 0;
        bool block = false;
        std::function<bool(fair::graph::tag_t)> trigger_predicate = {};
        gr::circular_buffer<T> buffer = gr::circular_buffer<T>(0);
        std::deque<pending_window_t> pending_trigger_windows; // triggers that still didn't receive all their data
        std::function<void(std::span<const T>)> callback = {}; // TODO we might want to pass back stats here like drop_count
        std::weak_ptr<dataset_poller> dataset_polling_handler = {};
        std::weak_ptr<poller> polling_handler = {};
        int64_t drop_count = 0;
    };

    std::vector<listener_t> listeners;
    std::mutex listener_mutex;

public:
    data_sink() {
        data_sink_registry::instance().register_sink(this);
    }

    ~data_sink() {
        data_sink_registry::instance().unregister_sink(this);
    }

    std::shared_ptr<poller> get_streaming_poller(blocking_mode block = blocking_mode::NonBlocking) {
        std::lock_guard lg(listener_mutex);
        auto handler = std::make_shared<poller>();
        listeners.push_back({
            .mode = acquisition_mode::Continuous,
            .block = block == blocking_mode::Blocking,
            .polling_handler = handler
        });
        return handler;
    }

    template<typename TriggerPredicate>
    std::shared_ptr<dataset_poller> get_trigger_poller(TriggerPredicate p, std::size_t pre_samples, std::size_t post_samples, blocking_mode block = blocking_mode::NonBlocking) {
        std::lock_guard lg(listener_mutex);
        auto handler = std::make_shared<dataset_poller>();
        listeners.push_back({
            .mode = acquisition_mode::Triggered,
            .pre_samples = pre_samples,
            .post_samples = post_samples,
            .block = block == blocking_mode::Blocking,
            .trigger_predicate = std::move(p),
            .dataset_polling_handler = handler
        });
        history.resize(std::max(pre_samples, history.size()));
        return handler;
    }

    template<typename Callback>
    void register_streaming_callback(Callback callback) {
        std::lock_guard lg(listener_mutex);
        listeners.push_back({
            .mode = acquisition_mode::Continuous,
            .callback = std::move(callback)
        });
    }

    // TODO this code should be called at the end of graph processing
    void stop() {
        std::lock_guard lg(listener_mutex);
        for (auto &listener : listeners) {
            if (listener.mode == acquisition_mode::Triggered || listener.mode == acquisition_mode::PostMortem) {
                // send out any incomplete data windows
                for (auto &window : listener.pending_trigger_windows) {
                    publish_dataset(listener, std::move(window.dataset));
                }

                if (auto p = listener.dataset_polling_handler.lock()) {
                    p->finished = true;
                }
            } else if (listener.mode == acquisition_mode::Continuous) {
                if (auto p = listener.polling_handler.lock()) {
                    // TODO pass remaining data
                    p->finished = true;
                }  else {
                    // TODO pass remaining data to callback
                }
            }
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
        assert(n_samples_consumed == std::round(in_data[0]));

        auto &tag_reader = in_port.tagReader();
        const auto n_tags = tag_reader.available();
        const auto tag_data = tag_reader.get(n_tags);
        std::vector<tag_t> tags(tag_data.begin(), tag_data.end());
        auto out_of_range = [end_pos = reader_position + noutput_items](const auto &tag) {
            return tag.index > end_pos;
        };
        std::erase_if(tags, out_of_range);
        tag_reader.consume(tags.size());

        {
            std::lock_guard lg(listener_mutex);
            for (auto &listener : listeners) {
                if (listener.mode == acquisition_mode::Continuous) {
                    write_to_listener(listener, std::vector<std::span<const T>>{in_data});
                }  else if (listener.mode == acquisition_mode::Triggered || listener.mode == acquisition_mode::PostMortem) {
                    namespace views = std::ranges::views;
                    auto filtered = tags | views::filter(listener.trigger_predicate);
                    for (const auto &trigger : filtered) {
                        // TODO fill dataset with metadata etc.
                        DataSet<T> dataset;
                        dataset.timing_events = {{trigger}};
                        dataset.signal_values.reserve(listener.pre_samples + listener.post_samples);
                        listener.pending_trigger_windows.push_back({.trigger = trigger, .dataset = std::move(dataset), .pending_post_samples = listener.post_samples});
                    }
                    auto window = listener.pending_trigger_windows.begin();
                    while (window != listener.pending_trigger_windows.end()) {
                        auto &dataset = window->dataset;
                        const auto window_offset = window->trigger.index - reader_position;

                        if (window_offset >= 0 && dataset.signal_values.empty()) { // new trigger, write history
                            // old history: pre-trigger data from previous in_data (if available)
                            const auto old_history_size = std::max(static_cast<std::int64_t>(listener.pre_samples) - window_offset, std::int64_t{0});
                            const auto available = std::min(static_cast<std::size_t>(old_history_size), history_view.size());
                            const auto old_history_view = history_view.last(available);
                            dataset.signal_values.insert(dataset.signal_values.end(), old_history_view.begin(), old_history_view.end());

                            // new history: pre-trigger samples from the current in_data
                            const auto new_history_size = listener.pre_samples - old_history_size;
                            const auto new_history_view = in_data.subspan(window_offset - new_history_size, new_history_size);
                            dataset.signal_values.insert(dataset.signal_values.end(), new_history_view.begin(), new_history_view.end());
                        }

                        // write missing post-samples
                        const auto previous_post_samples = listener.post_samples - window->pending_post_samples;
                        const auto first_requested = window_offset + previous_post_samples;
                        const auto last_requested = window_offset + listener.post_samples - 1;
                        const auto last_available = std::min(last_requested, noutput_items - 1);
                        const auto post_sample_view = in_data.subspan(first_requested, last_available - first_requested + 1);
                        dataset.signal_values.insert(dataset.signal_values.end(), post_sample_view.begin(), post_sample_view.end());
                        window->pending_post_samples -= post_sample_view.size();

                        if (window->pending_post_samples == 0) {
                            publish_dataset(listener, std::move(dataset));
                            window = listener.pending_trigger_windows.erase(window);
                        } else {
                            ++window;
                        }
                    }
                }
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

    inline void publish_dataset(listener_t &l, DataSet<T> &&data) {
        if (auto poller = l.dataset_polling_handler.lock()) {
            auto write_data = poller->writer.reserve_output_range(1);
            if (l.block) {
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
        } else { // if callback...
            // TODO call callback
        }
    }

    inline void write_to_listener(listener_t &l, const std::vector<std::span<const T>> &spans) {
        std::size_t total_size = 0;
        for (const auto &i : spans) {
            total_size += i.size();
        }

        if (total_size == 0) {
            return;
        }

        if (auto poller = l.polling_handler.lock()) {
            auto writer = poller->buffer.new_writer();
            if (l.block) {
                auto write_data = writer.reserve_output_range(total_size);
                auto target = write_data.begin();
                for (const auto &in_data : spans) {
                    std::copy(in_data.begin(), in_data.end(), target);
                    target += in_data.size();
                }
                write_data.publish(write_data.size());
            } else {
                const auto can_write = writer.available();
                auto to_write = std::min(total_size, can_write);
                poller->drop_count += total_size - can_write;
                if (to_write > 0) {
                    auto write_data = writer.reserve_output_range(to_write);
                    std::size_t written = 0;
                    for (const auto &in_data : spans) {
                        const auto n = std::min(in_data.size(), to_write - written);
                        std::copy(in_data.begin(), in_data.begin() + n, write_data.begin() + written);
                        written += n;
                        if (written == to_write) {
                            break;
                        }
                    }
                    write_data.publish(write_data.size());
                }
            }
        } else if (l.callback) {
            // TODO use buffer/make this one call
            for (const auto &data : spans) {
                l.callback(data);
            }
        }
    }
};

}

ENABLE_REFLECTION_FOR_TEMPLATE_FULL((typename T), (fair::graph::data_sink<T>), in, n_samples_consumed, n_samples_max, last_tag_position, sample_rate);

#endif
