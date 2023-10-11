#ifndef GNURADIO_DATASET_HPP
#define GNURADIO_DATASET_HPP

#include "reflection.hpp"
#include "Tag.hpp"
#include <chrono>
#include <cstdint>
#include <map>
#include <pmtv/pmt.hpp>
#include <variant>
#include <vector>

namespace gr {

struct layout_right {};

struct layout_left {};

/**
 * @brief a concept that describes a Packet, which is a subset of the DataSet struct.
 */
template<typename T>
concept PacketLike = requires(T t, const std::size_t n_items) {
    typename T::value_type;
    typename T::pmt_map;
    std::is_same_v<decltype(t.timestamp), int64_t>;
    std::is_same_v<decltype(t.signal_values), std::vector<typename T::value_type>>;
    std::is_same_v<decltype(t.meta_information), std::vector<typename T::pmt_map>>;
};

/**
 * @brief A concept that describes a Tensor, which is a subset of the DataSet struct.
 */
template<typename T>
concept TensorLike = PacketLike<T> && requires(T t, const std::size_t n_items) {
    typename T::value_type;
    typename T::pmt_map;
    typename T::tensor_layout_type;
    std::is_same_v<decltype(t.extents), std::vector<std::int32_t>>;
    std::is_same_v<decltype(t.layout), std::vector<typename T::tensor_layout_type>>;
    std::is_same_v<decltype(t.signal_values), std::vector<typename T::value_type>>;
    std::is_same_v<decltype(t.signal_errors), std::vector<typename T::value_type>>;
    std::is_same_v<decltype(t.meta_information), std::vector<typename T::pmt_map>>;
};

/**
 * @brief: a DataSet consists of signal data, metadata, and associated axis information.
 *
 * The DataSet can be used to store and manipulate data in a structured way, and supports various types of axes,
 * layouts, and signal data. The dataset contains information such as timestamp, axis names and units, signal names,
 * values, and ranges, as well as metadata and timing events. This struct provides a flexible way to store and organize
 * data with associated metadata, and can be customized for different types of data and applications.
 */
template<typename T>
concept DataSetLike = TensorLike<T> && requires(T t, const std::size_t n_items) {
    typename T::value_type;
    typename T::pmt_map;
    typename T::tensor_layout_type;
    std::is_same_v<decltype(t.timestamp), int64_t>;

    // axis layout:
    std::is_same_v<decltype(t.axis_names), std::vector<std::string>>;
    std::is_same_v<decltype(t.axis_units), std::vector<std::string>>;
    std::is_same_v<decltype(t.axis_values), std::vector<typename T::value_type>>;

    // signal data storage
    std::is_same_v<decltype(t.signal_names), std::vector<std::string>>;
    std::is_same_v<decltype(t.signal_units), std::vector<std::string>>;
    std::is_same_v<decltype(t.signal_values), std::vector<typename T::value_type>>;
    std::is_same_v<decltype(t.signal_errors), std::vector<typename T::value_type>>;
    std::is_same_v<decltype(t.signal_ranges), std::vector<std::vector<typename T::value_type>>>;

    // meta data
    std::is_same_v<decltype(t.meta_information), std::vector<typename T::pmt_map>>;
    std::is_same_v<decltype(t.timing_events), std::vector<std::vector<Tag>>>;
};

template<typename T>
struct DataSet {
    using value_type         = T;
    using tensor_layout_type = std::variant<layout_right, layout_left, std::string>;
    using pmt_map            = std::map<std::string, pmtv::pmt>;
    std::int64_t timestamp   = 0; // UTC timestamp [ns]

    // axis layout:
    std::vector<std::string>    axis_names;  // e.g. time, frequency, …
    std::vector<std::string>    axis_units;  // axis base SI-unit
    std::vector<std::vector<T>> axis_values; // explicit axis values

    // signal data layout:
    std::vector<std::int32_t> extents; // extents[dim0_size, dim1_size, …]
    tensor_layout_type        layout;  // row-major, column-major, “special”

    // signal data storage:
    std::vector<std::string>    signal_names;  // size = extents[0]
    std::vector<std::string>    signal_units;  // size = extents[0]
    std::vector<T>              signal_values; // size = \PI_i extents[i]
    std::vector<T>              signal_errors; // size = \PI_i extents[i] or '0' if not applicable
    std::vector<std::vector<T>> signal_ranges; // [[min_0, max_0], [min_1, max_1], …] used for communicating, for example, HW limits

    // meta data
    std::vector<pmt_map>          meta_information;
    std::vector<std::vector<Tag>> timing_events;
};

static_assert(DataSetLike<DataSet<std::byte>>, "DataSet<std::byte> concept conformity");
static_assert(DataSetLike<DataSet<float>>, "DataSet<float> concept conformity");
static_assert(DataSetLike<DataSet<double>>, "DataSet<double> concept conformity");

// public type definitions to allow simple reflection
using DataSet_float  = DataSet<double>;
using DataSet_double = DataSet<float>;

template<typename T>
struct Tensor {
    using value_type                    = T;
    using tensor_layout_type            = std::variant<layout_right, layout_left, std::string>;
    using pmt_map                       = std::map<std::string, pmtv::pmt>;
    std::int64_t              timestamp = 0; // UTC timestamp [ns]

    std::vector<std::int32_t> extents; // extents[dim0_size, dim1_size, …]
    tensor_layout_type        layout;  // row-major, column-major, “special”

    std::vector<T>            signal_values; // size = \PI_i extents[i]
    std::vector<T>            signal_errors; // size = \PI_i extents[i] or '0' if not applicable

    // meta data
    std::vector<pmt_map> meta_information;
};

static_assert(TensorLike<Tensor<std::byte>>, "Tensor<std::byte> concept conformity");
static_assert(TensorLike<Tensor<float>>, "Tensor<std::byte> concept conformity");
static_assert(TensorLike<Tensor<double>>, "Tensor<std::byte> concept conformity");

template<typename T>
struct Packet {
    using value_type               = T;
    using pmt_map                  = std::map<std::string, pmtv::pmt>;

    std::int64_t         timestamp = 0; // UTC timestamp [ns]
    std::vector<T>       signal_values; // size = \PI_i extents[i
    std::vector<pmt_map> meta_information;
};

static_assert(PacketLike<Packet<std::byte>>, "Packet<std::byte> concept conformity");
static_assert(PacketLike<Packet<float>>, "Packet<std::byte> concept conformity");
static_assert(PacketLike<Packet<double>>, "Packet<std::byte> concept conformity");

} // namespace gr

ENABLE_REFLECTION(gr::DataSet_double, timestamp, axis_names, axis_units, axis_values, extents, layout, signal_names, signal_units, signal_values, signal_errors, signal_ranges, meta_information,
                  timing_events)
ENABLE_REFLECTION(gr::DataSet_float, timestamp, axis_names, axis_units, axis_values, extents, layout, signal_names, signal_units, signal_values, signal_errors, signal_ranges, meta_information,
                  timing_events)
#endif // GNURADIO_DATASET_HPP
