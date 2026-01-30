#ifndef GNURADIO_DATASET_HPP
#define GNURADIO_DATASET_HPP

#include <chrono>
#include <cstdint>
#include <map>
#include <variant>
#include <vector>

#include <gnuradio-4.0/meta/reflection.hpp>

#include "Message.hpp"
#include "Tag.hpp"

namespace gr {

struct LayoutRight {};

struct LayoutLeft {};

template<typename T>
struct Range {
    T min = 0;
    T max = 0;
    GR_MAKE_REFLECTABLE(Range, min, max);

    auto operator<=>(const Range<T>& other) const = default;
};

/**
 * @brief a concept that describes a Packet, which is a subset of the DataSet struct.
 */
template<typename U, typename T = std::remove_cvref_t<U>>
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
template<typename U, typename T = std::remove_cvref_t<U>>
concept TensorLikeV2 = PacketLike<T> && requires(T t, const std::size_t n_items) {
    typename T::value_type;
    typename T::pmt_map;
    typename T::tensor_layout_type;
    requires std::is_same_v<decltype(t.extents), std::vector<std::int32_t>>;
    requires std::is_same_v<decltype(t.layout), typename T::tensor_layout_type>;
    requires std::is_same_v<decltype(t.signal_values), std::vector<typename T::value_type>>;
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
template<typename U, typename T = std::remove_cvref_t<U>>
concept DataSetLike = TensorLikeV2<T> && requires(T t, const std::size_t n_items) {
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
    requires std::is_same_v<decltype(t.signal_ranges), std::vector<Range<typename T::value_type>>>;

    // meta data
    requires std::is_same_v<decltype(t.meta_information), std::vector<typename T::pmt_map>>;
    requires std::is_same_v<decltype(t.timing_events), std::vector<std::vector<std::pair<std::ptrdiff_t, gr::pmt::Value::Map>>>>;
};

template<typename T>
struct DataSet {
    using value_type           = T;
    using tensor_layout_type   = std::variant<LayoutRight, LayoutLeft, std::string>;
    using pmt_map              = gr::property_map;
    using idx_pmt_map          = std::pair<std::ptrdiff_t, pmt_map>;
    T            default_value = T(); // default value for padding, ZOH etc.
    std::int64_t timestamp     = 0;   // UTC timestamp [ns]

    // axis layout:
    std::vector<std::string>    axis_names{};  // axis quantity, e.g. time, frequency, …
    std::vector<std::string>    axis_units{};  // axis base SI-unit
    std::vector<std::vector<T>> axis_values{}; // explicit axis values

    // signal data layout:
    std::vector<std::int32_t> extents{}; // extents[dim0_size, dim1_size, …] i.e. [axis_values[0].size(), axis_values[1].size(), …]
    tensor_layout_type        layout{};  // row-major, column-major, “special”

    // signal data storage:
    std::vector<std::string> signal_names{};      // defines number of signals, i.e. 'this->size()'
    std::vector<std::string> signal_quantities{}; // size = this->size()
    std::vector<std::string> signal_units{};      // size = this->size()
    std::vector<T>           signal_values{};     // size = this->size() × Π_i extents[i]
    std::vector<Range<T>>    signal_ranges{};     // [[min_0, max_0], [min_1, max_1], …] used for communicating, for example, HW limits

    // meta data
    std::vector<pmt_map>                  meta_information{};
    std::vector<std::vector<idx_pmt_map>> timing_events{};

    GR_MAKE_REFLECTABLE(DataSet, timestamp, axis_names, axis_units, axis_values, extents, layout, signal_names, signal_quantities, signal_units, signal_values, signal_ranges, meta_information, timing_events);

    [[nodiscard]] std::size_t nDimensions() const noexcept { return extents.size(); }

    [[nodiscard]] std::size_t        axisCount() const noexcept { return axis_names.size(); }
    [[nodiscard]] std::string&       axisName(std::size_t axisIdx = 0UZ) { return axis_names[_axCheck(axisIdx)]; }
    [[nodiscard]] std::string_view   axisName(std::size_t axisIdx = 0UZ) const { return axis_names[_axCheck(axisIdx)]; }
    [[nodiscard]] std::string&       axisUnit(std::size_t axisIdx = 0UZ) { return axis_units[_axCheck(axisIdx)]; }
    [[nodiscard]] std::string_view   axisUnit(std::size_t axisIdx = 0UZ) const { return axis_units[_axCheck(axisIdx)]; }
    [[nodiscard]] std::span<T>       axisValues(std::size_t axisIdx = 0UZ) { return axis_values[_axCheck(axisIdx)]; }
    [[nodiscard]] std::span<const T> axisValues(std::size_t axisIdx = 0UZ) const { return axis_values[_axCheck(axisIdx)]; }

    [[nodiscard]] constexpr std::size_t size() const noexcept { return signal_names.size(); }
    [[nodiscard]] std::string&          signalName(std::size_t signalIdx = 0UZ) { return signal_names[_idxCheck(signalIdx)]; }
    [[nodiscard]] std::string_view      signalName(std::size_t signalIdx = 0UZ) const { return signal_names[_idxCheck(signalIdx)]; }
    [[nodiscard]] std::string&          signalQuantity(std::size_t signalIdx = 0UZ) { return signal_quantities[_idxCheck(signalIdx)]; }
    [[nodiscard]] std::string_view      signalQuantity(std::size_t signalIdx = 0UZ) const { return signal_quantities[_idxCheck(signalIdx)]; }
    [[nodiscard]] std::string&          signalUnit(std::size_t signalIdx = 0UZ) { return signal_units[_idxCheck(signalIdx)]; }
    [[nodiscard]] std::string_view      signalUnit(std::size_t signalIdx = 0UZ) const { return signal_units[_idxCheck(signalIdx)]; }
    [[nodiscard]] std::span<T>          signalValues(std::size_t signalIdx = 0UZ) { return {std::next(signal_values.data(), _idxCheckS(signalIdx) * _valsPerSigS()), _valsPerSig()}; }
    [[nodiscard]] std::span<const T>    signalValues(std::size_t signalIdx = 0UZ) const { return {std::next(signal_values.data(), _idxCheckS(signalIdx) * _valsPerSigS()), _valsPerSig()}; }
    [[nodiscard]] Range<T>&             signalRange(std::size_t signalIdx = 0UZ) { return signal_ranges[_idxCheck(signalIdx)]; }
    [[nodiscard]] const Range<T>&       signalRange(std::size_t signalIdx = 0UZ) const { return signal_ranges[_idxCheck(signalIdx)]; }

    [[nodiscard]] pmt_map&                     metaInformation(std::size_t signalIdx = 0UZ) { return meta_information[_idxCheck(signalIdx)]; }
    [[nodiscard]] const pmt_map&               metaInformation(std::size_t signalIdx = 0UZ) const { return meta_information[_idxCheck(signalIdx)]; }
    [[nodiscard]] std::span<idx_pmt_map>       timingEvents(std::size_t signalIdx = 0UZ) { return timing_events[_idxCheck(signalIdx)]; }
    [[nodiscard]] std::span<const idx_pmt_map> timingEvents(std::size_t signalIdx = 0UZ) const { return timing_events[_idxCheck(signalIdx)]; }

private:
    [[nodiscard]] std::size_t _axCheck(std::size_t i, std::source_location loc = std::source_location::current()) const {
        if (i >= axis_names.size()) {
            throw gr::exception(std::format("{} axis out of range: i={} >= axis_name [0, {}]", loc.function_name(), i, axis_names.size()), loc);
        }
        if (i >= axis_values.size()) {
            throw gr::exception(std::format("{} axis out of range: i={} >= axis_values [0, {}]", loc.function_name(), i, axis_values.size()), loc);
        }
        return i;
    }

    [[nodiscard]] std::size_t _idxCheck(std::size_t i, std::source_location location = std::source_location::current()) const {
        if (i >= size()) {
            throw gr::exception(std::format("{} out of range: i={} >= [0, {}]", location.function_name(), i, size()), location);
        }
        return i;
    }

    [[nodiscard]] std::ptrdiff_t _idxCheckS(std::size_t i, std::source_location location = std::source_location::current()) const {
        if (i >= size()) {
            throw gr::exception(std::format("{} out of range: i={} >= [0, {}]", location.function_name(), i, size()), location);
        }
        return static_cast<std::ptrdiff_t>(i);
    }

    [[nodiscard]] std::size_t    _valsPerSig() const noexcept { return size() == 0U ? 0U : signal_values.size() / size(); }
    [[nodiscard]] std::ptrdiff_t _valsPerSigS() const noexcept { return static_cast<std::ptrdiff_t>(size() == 0U ? 0U : signal_values.size() / size()); }
};

static_assert(DataSetLike<DataSet<std::byte>>, "DataSet<std::byte> concept conformity");
static_assert(DataSetLike<DataSet<float>>, "DataSet<float> concept conformity");
static_assert(DataSetLike<DataSet<double>>, "DataSet<double> concept conformity");

template<typename T>
struct Packet {
    using value_type = T;
    using pmt_map    = pmt::Value::Map;
    T default_value  = T(); // default value for padding, ZOH etc.

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
