#ifndef GRAPH_PROTOTYPE_DATASET_HPP
#define GRAPH_PROTOTYPE_DATASET_HPP

#include <vector>
#include <cstdint>
#include <chrono>
#include <map>
#include <pmtv/pmt.hpp>
#include <node.hpp>

namespace graph::dataset {
using namespace std::chrono_literals;

/**
 * n-dimensional tensor
 * corresponds to DataSet layout
 * uses signed integers for size type for Java interoperability
 * dense data on mesh
 * - concatenation of datasets -> list of datasets or copy into single dataset
 * to be discussed:
 * - general layout
 * - axis values: min/max -> equidistant or grid, flattened or vec<vec>
 * - signal dimension? mesh vs point cloud
 * - vector of map? other representations
 * - layout policy: mdspan, strides, other solutions?
 */
template<typename T, class Allocator = std::allocator<T>>
class DataSet {
public:
    template<typename R>
    using vector = std::vector<R>;//, Allocator<R>>;
    using status_map = std::vector<pmtv::map_t>;
    using timing_map = std::vector<std::map<int64_t,pmtv::pmt>>;

    std::int64_t         timestamp = 0; // [ns] "UTC timestamp on which the timing event occurred"
    vector<std::int32_t> extents;       // size equal to rank+1, entries are size of individual dimensions // "extents of the different dimensions"
    // todo: layout_policy/strides
    vector<std::string>  axisNames;     // size equals rank // "e.g. time, frequency, voltage, current"
    vector<std::string>  axisUnits;     // size equals rank // "base si-unit for axis"
    vector<T>            axisValues;    // TODO: nested(outer size = rank, inner size = extents[i] or 2) or flattended? min/max acq. range (ADC clamping, THD, ...) // flattened because of serialiser limitations // "explicit axis values, not necessarily equidistant"
    // signalDimension;                 // size = extents[0], 0->xAxis, 1-> yAxis, ... // needs further investigation how to implement
    vector<std::string>  signalNames;   // size = extents[0] // "name of the signal"
    vector<std::string>  signalUnits;   // size = extents[0] // "base si-units"
    vector<T>            signalValues;  // actual signal data, size = \PI_i extents[i] // "values"
    vector<T>            signalErrors;  // actual signal errors, size = \PI_i extents[i] or 0 TODO: model errors as extra dimension instead of separate field? // "rms errors"
    vector<T>            signalRanges;  // size = extents[0] * 2 [min_0, max_0, min_1, ...] // "min/max value of signal"
    status_map           signalStatus;  // "status messages for signal"
    timing_map           timingEvents;  // "raw timing events occurred in the acq window"
};
// public type definitions to allow simple reflection
using DataSet_float = DataSet<double>;
using DataSet_double = DataSet<float>;
} // graph:dataset

ENABLE_REFLECTION(graph::dataset::DataSet_double, timestamp, signalNames, axisUnits, axisNames, axisValues, extents, signalValues, signalErrors, signalStatus, timingEvents)
ENABLE_REFLECTION(graph::dataset::DataSet_float, timestamp, signalNames, axisUnits, axisNames, axisValues, extents, signalValues, signalErrors, signalStatus, timingEvents)
#endif //GRAPH_PROTOTYPE_DATASET_HPP
