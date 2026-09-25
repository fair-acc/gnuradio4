#ifndef GNURADIO_TESTING_EVENT_SCRIPT_HPP
#define GNURADIO_TESTING_EVENT_SCRIPT_HPP

#include <cstddef>
#include <utility>
#include <vector>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/DataSet.hpp>
#include <gnuradio-4.0/Port.hpp>

namespace gr::testing {

/** publishes the fixed list `_events`, so which of them a block acts on does not depend on work-call granularity */
struct EventScript : gr::Block<EventScript> {
    /** an event output claims one slot per work call unless told otherwise */
    gr::EventPortOut evtOut{{.streamSlotsPerPublish = 8UZ}};

    GR_MAKE_REFLECTABLE(EventScript, evtOut);

    std::vector<gr::property_map> _events;
    std::size_t                   _published = 0UZ;

    gr::work::Status processBulk(gr::OutputSpanLike auto& evtSpan) {
        std::size_t emitted = 0UZ;
        while (_published < _events.size() && emitted < evtSpan.size()) {
            if (!gr::emit(evtSpan, emitted, gr::property_map_view{_events[_published]})) {
                break;
            }
            ++_published;
            ++emitted;
        }
        evtSpan.publish(emitted);
        return gr::work::Status::OK;
    }
};

template<typename T>
struct DataSetCollector : gr::Block<DataSetCollector<T>> {
    gr::PortIn<gr::DataSet<T>> in;

    GR_MAKE_REFLECTABLE(DataSetCollector, in);

    std::vector<gr::DataSet<T>> _sets;

    void processOne(gr::DataSet<T> set) { _sets.push_back(std::move(set)); }
};

} // namespace gr::testing

#endif // GNURADIO_TESTING_EVENT_SCRIPT_HPP
