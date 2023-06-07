#ifndef GNURADIO_DATA_SINK_HPP
#define GNURADIO_DATA_SINK_HPP

#include "circular_buffer.hpp"
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

private:
    struct listener {
        acquisition_mode mode = acquisition_mode::Triggered;
        std::pair<int64_t, int64_t> window; ///< window of data to return in relation to the matching tag position, e.g. [-2000, 3000] to obtain 2000 presamples and 3000 postsamples
        std::size_t history_size = 0;
        bool block = false;
        int64_t drop_count = 0;
        std::function<bool(fair::graph::tag_t::map_type)> trigger_predicate;
        gr::circular_buffer<T> buffer;
        std::optional<std::size_t> pending; ///< number of samples expected due to a previous trigger
        std::function<void(std::span<const T>)> callback; // TODO we might want to pass back stats here like drop_count
        std::weak_ptr<poller> polling_handler;
    };

    struct {
        std::atomic<bool> dirty = false;
        std::mutex mutex;
        std::vector<listener> list;
    } pending_listeners;

    std::vector<listener> listeners;

public:
    // TODO sink should register itself on construction, but name is set afterwards via
    // set_name, which we have no hook into. Maybe the registration should be done by the
    // graph creating/destroying the sink instead?

    data_sink() {
        data_sink_registry::instance().register_sink(this);
    }

    ~data_sink() {
        data_sink_registry::instance().unregister_sink(this);
    }

    std::shared_ptr<poller> get_streaming_poller(blocking_mode block = blocking_mode::NonBlocking) {
        auto handler = std::make_shared<poller>();
        pending_listeners.list.push_back({
            .mode = acquisition_mode::Continuous,
            .block = block == blocking_mode::Blocking,
            .buffer = gr::circular_buffer<T>(0),
            .polling_handler = handler
        });
        pending_listeners.dirty = true;
        return handler;
    }

    template<typename Callback>
    void register_streaming_callback(Callback callback) {
        std::lock_guard lg(pending_listeners.mutex);
        pending_listeners.list.push_back({
            .mode = acquisition_mode::Continuous,
            .buffer = gr::circular_buffer<T>(0),
            .callback = std::move(callback)
        });
        pending_listeners.dirty = true;
    }

    [[nodiscard]] work_return_t work() {
        auto &in_port = input_port<"in">(this);
        auto &reader = in_port.streamReader();

        const auto n_readable = std::min(reader.available(), in_port.max_buffer_size());
        if (n_readable == 0) {
            return fair::graph::work_return_t::INSUFFICIENT_INPUT_ITEMS;
        }

        const auto noutput_items = std::min(listener_buffer_size, n_readable);
        const auto in_data = reader.get(noutput_items);

        if (pending_listeners.dirty) {
            std::lock_guard lg(pending_listeners.mutex);
            listeners = pending_listeners.list;
            pending_listeners.dirty = false;
        }

        for (auto &listener : listeners) {
            if (listener.mode == acquisition_mode::Continuous) {
                if (auto poller = listener.polling_handler.lock()) {
                    auto writer = poller->buffer.new_writer();
                    const auto read_data = reader.get(noutput_items);
                    if (listener.block) {
                        auto write_data = writer.reserve_output_range(noutput_items);
                        std::copy(read_data.begin(), read_data.end(), write_data.begin());
                        write_data.publish(write_data.size());
                    } else {
                        const auto can_write = writer.available();
                        const auto to_write = std::min(read_data.size(), can_write);
                        poller->drop_count += read_data.size() - can_write;
                        if (to_write > 0) {
                            auto write_data = writer.reserve_output_range(to_write);
                            std::copy(read_data.begin(), read_data.begin() + to_write - 1, write_data.begin());
                            write_data.publish(write_data.size());
                        }
                    }
                } else if (listener.callback) {
                    listener.callback(in_data);
                }
            }
        }

        n_samples_consumed += noutput_items;

        if (!reader.consume(noutput_items)) {
            return work_return_t::ERROR;
        }

        return work_return_t::OK;
    }
};

}

ENABLE_REFLECTION_FOR_TEMPLATE_FULL((typename T), (fair::graph::data_sink<T>), in, n_samples_consumed, n_samples_max, last_tag_position, sample_rate);

#endif
