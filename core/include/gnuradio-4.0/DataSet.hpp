#ifndef GNURADIO_DATASET_HPP
#define GNURADIO_DATASET_HPP

#include "Tag.hpp"
#include <chrono>
#include <cstdint>
#include <gnuradio-4.0/meta/reflection.hpp>
#include <map>
#include <pmtv/pmt.hpp>
#include <variant>
#include <vector>

namespace gr {

struct LayoutRight {};

struct LayoutLeft {};

/**
 * @brief a concept that describes a Packet, which is a subset of the DataSet struct.
 */
template<typename T>
concept PacketLike = requires(T t) {
    typename T::value_type;
    typename T::pmt_map;
    requires std::is_same_v<decltype(t.timestamp), int64_t>;
    requires std::is_same_v<decltype(t.signal_values), std::vector<typename T::value_type>>;
    requires std::is_same_v<decltype(t.meta_information), std::vector<typename T::pmt_map>>;
};

/**
 * @brief A concept that describes a Tensor, which is a subset of the DataSet struct.
 */
template<typename T>
concept TensorLike = PacketLike<T> && requires(T t, const std::size_t n_items) {
    typename T::value_type;
    typename T::pmt_map;
    typename T::tensor_layout_type;
    requires std::is_same_v<decltype(t.extents), std::vector<std::int32_t>>;
    requires std::is_same_v<decltype(t.layout), typename T::tensor_layout_type>;
    requires std::is_same_v<decltype(t.signal_values), std::vector<typename T::value_type>>;
    requires std::is_same_v<decltype(t.signal_errors), std::vector<typename T::value_type>>;
    requires std::is_same_v<decltype(t.meta_information), std::vector<typename T::pmt_map>>;
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
    requires std::is_same_v<decltype(t.timestamp), int64_t>;

    // axis layout:
    requires std::is_same_v<decltype(t.axis_names), std::vector<std::string>>;
    requires std::is_same_v<decltype(t.axis_units), std::vector<std::string>>;
    requires std::is_same_v<decltype(t.axis_values), std::vector<std::vector<typename T::value_type>>>;

    // signal data storage
    requires std::is_same_v<decltype(t.signal_names), std::vector<std::string>>;
    requires std::is_same_v<decltype(t.signal_quantities), std::vector<std::string>>;
    requires std::is_same_v<decltype(t.signal_units), std::vector<std::string>>;
    requires std::is_same_v<decltype(t.signal_values), std::vector<typename T::value_type>>;
    requires std::is_same_v<decltype(t.signal_errors), std::vector<typename T::value_type>>;
    requires std::is_same_v<decltype(t.signal_ranges), std::vector<std::vector<typename T::value_type>>>;

    // meta data
    requires std::is_same_v<decltype(t.meta_information), std::vector<typename T::pmt_map>>;
    requires std::is_same_v<decltype(t.timing_events), std::vector<std::vector<std::pair<std::ptrdiff_t, property_map>>>>;
};

template<typename T>
struct DataSet {
    using value_type         = T;
    using tensor_layout_type = std::variant<LayoutRight, LayoutLeft, std::string>;
    using pmt_map            = std::map<std::string, pmtv::pmt>;
    std::int64_t timestamp   = 0; // UTC timestamp [ns]

    // axis layout:
    std::vector<std::string>    axis_names{};  // e.g. time, frequency, …
    std::vector<std::string>    axis_units{};  // axis base SI-unit
    std::vector<std::vector<T>> axis_values{}; // explicit axis values

    // signal data layout:
    std::vector<std::int32_t> extents{}; // extents[dim0_size, dim1_size, …]
    tensor_layout_type        layout{};  // row-major, column-major, “special”

    // signal data storage:
    std::vector<std::string>    signal_names{};      // size = extents[0]
    std::vector<std::string>    signal_quantities{}; // size = extents[0]
    std::vector<std::string>    signal_units{};      // size = extents[0]
    std::vector<T>              signal_values{};     // size = \PI_i extents[i]
    std::vector<T>              signal_errors{};     // size = \PI_i extents[i] or '0' if not applicable
    std::vector<std::vector<T>> signal_ranges{};     // [[min_0, max_0], [min_1, max_1], …] used for communicating, for example, HW limits

    // meta data
    std::vector<pmt_map>                                              meta_information{};
    std::vector<std::vector<std::pair<std::ptrdiff_t, property_map>>> timing_events{};

    GR_MAKE_REFLECTABLE(DataSet, timestamp, axis_names, axis_units, axis_values, extents, layout, signal_names, signal_quantities, signal_units, signal_values, signal_errors, signal_ranges, meta_information, timing_events);
};

static_assert(DataSetLike<DataSet<std::byte>>, "DataSet<std::byte> concept conformity");
static_assert(DataSetLike<DataSet<float>>, "DataSet<float> concept conformity");
static_assert(DataSetLike<DataSet<double>>, "DataSet<double> concept conformity");

template<typename T>
struct Tensor {
    using value_type         = T;
    using tensor_layout_type = std::variant<LayoutRight, LayoutLeft, std::string>;
    using pmt_map            = std::map<std::string, pmtv::pmt>;
    std::int64_t timestamp   = 0; // UTC timestamp [ns]

    std::vector<std::int32_t> extents{}; // extents[dim0_size, dim1_size, …]
    tensor_layout_type        layout{};  // row-major, column-major, “special”

    std::vector<T> signal_values{}; // size = \PI_i extents[i]
    std::vector<T> signal_errors{}; // size = \PI_i extents[i] or '0' if not applicable

    // meta data
    std::vector<pmt_map> meta_information{};

    GR_MAKE_REFLECTABLE(Tensor, timestamp, extents, layout, signal_values, signal_errors, meta_information);
};

static_assert(TensorLike<Tensor<std::byte>>, "Tensor<std::byte> concept conformity");
static_assert(TensorLike<Tensor<float>>, "Tensor<std::byte> concept conformity");
static_assert(TensorLike<Tensor<double>>, "Tensor<std::byte> concept conformity");

template<typename T>
struct Packet {
    using value_type = T;
    using pmt_map    = std::map<std::string, pmtv::pmt>;

    std::int64_t         timestamp = 0;   // UTC timestamp [ns]
    std::vector<T>       signal_values{}; // size = \PI_i extents[i
    std::vector<pmt_map> meta_information{};

    GR_MAKE_REFLECTABLE(Packet, timestamp, signal_values, meta_information);
};

static_assert(PacketLike<Packet<std::byte>>, "Packet<std::byte> concept conformity");
static_assert(PacketLike<Packet<float>>, "Packet<std::byte> concept conformity");
static_assert(PacketLike<Packet<double>>, "Packet<std::byte> concept conformity");

} // namespace gr

#endif // GNURADIO_DATASET_HPP
