#ifndef GNURADIO_DATASET_HPP
#define GNURADIO_DATASET_HPP

#include <chrono>
#include <cstdint>
#include <map>
#include <memory_resource>
#include <string>
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
    requires std::is_same_v<decltype(t.signal_values), std::pmr::vector<typename T::value_type>>;
    requires std::is_same_v<decltype(t.meta_information), std::pmr::vector<typename T::pmt_map>>;
};

/**
 * @brief A concept that describes a Tensor, which is a subset of the DataSet struct.
 */
template<typename U, typename T = std::remove_cvref_t<U>>
concept TensorLikeV2 = PacketLike<T> && requires(T t, const std::size_t n_items) {
    typename T::value_type;
    typename T::pmt_map;
    typename T::tensor_layout_type;
    requires std::is_same_v<decltype(t.extents), std::pmr::vector<std::int32_t>>;
    requires std::is_same_v<decltype(t.layout), typename T::tensor_layout_type>;
    requires std::is_same_v<decltype(t.signal_values), std::pmr::vector<typename T::value_type>>;
    requires std::is_same_v<decltype(t.meta_information), std::pmr::vector<typename T::pmt_map>>;
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
    requires std::is_same_v<decltype(t.axis_names), std::pmr::vector<std::pmr::string>>;
    requires std::is_same_v<decltype(t.axis_units), std::pmr::vector<std::pmr::string>>;
    requires std::is_same_v<decltype(t.axis_values), std::pmr::vector<std::pmr::vector<typename T::value_type>>>;

    // signal data storage
    requires std::is_same_v<decltype(t.signal_names), std::pmr::vector<std::pmr::string>>;
    requires std::is_same_v<decltype(t.signal_quantities), std::pmr::vector<std::pmr::string>>;
    requires std::is_same_v<decltype(t.signal_units), std::pmr::vector<std::pmr::string>>;
    requires std::is_same_v<decltype(t.signal_values), std::pmr::vector<typename T::value_type>>;
    requires std::is_same_v<decltype(t.signal_ranges), std::pmr::vector<Range<typename T::value_type>>>;

    // meta data
    requires std::is_same_v<decltype(t.meta_information), std::pmr::vector<typename T::pmt_map>>;
    requires std::is_same_v<decltype(t.timing_events), std::pmr::vector<std::pmr::vector<std::pair<std::ptrdiff_t, gr::property_map>>>>;
};

template<typename T>
struct DataSet {
    using value_type           = T;
    using allocator_type       = std::pmr::polymorphic_allocator<>;
    using tensor_layout_type   = std::variant<LayoutRight, LayoutLeft, std::string>;
    using pmt_map              = gr::property_map;
    using idx_pmt_map          = std::pair<std::ptrdiff_t, pmt_map>;
    T            default_value = T(); // default value for padding, ZOH etc.
    std::int64_t timestamp     = 0;   // UTC timestamp [ns]

    // axis layout:
    std::pmr::vector<std::pmr::string>    axis_names{};  // axis quantity, e.g. time, frequency, …
    std::pmr::vector<std::pmr::string>    axis_units{};  // axis base SI-unit
    std::pmr::vector<std::pmr::vector<T>> axis_values{}; // explicit axis values

    // signal data layout:
    std::pmr::vector<std::int32_t> extents{}; // extents[dim0_size, dim1_size, …] i.e. [axis_values[0].size(), axis_values[1].size(), …]
    tensor_layout_type             layout{};  // row-major, column-major, “special”

    // signal data storage:
    std::pmr::vector<std::pmr::string> signal_names{};      // defines number of signals, i.e. 'this->size()'
    std::pmr::vector<std::pmr::string> signal_quantities{}; // size = this->size()
    std::pmr::vector<std::pmr::string> signal_units{};      // size = this->size()
    std::pmr::vector<T>                signal_values{};     // size = this->size() × Π_i extents[i]
    std::pmr::vector<Range<T>>         signal_ranges{};     // [[min_0, max_0], [min_1, max_1], …] used for communicating, for example, HW limits

    // meta data
    std::pmr::vector<pmt_map>                       meta_information{};
    std::pmr::vector<std::pmr::vector<idx_pmt_map>> timing_events{};

    GR_MAKE_REFLECTABLE(DataSet, default_value, timestamp, axis_names, axis_units, axis_values, extents, layout, signal_names, signal_quantities, signal_units, signal_values, signal_ranges, meta_information, timing_events);

    DataSet() = default;
    explicit DataSet(const allocator_type& alloc) : axis_names(alloc), axis_units(alloc), axis_values(alloc), extents(alloc), signal_names(alloc), signal_quantities(alloc), signal_units(alloc), signal_values(alloc), signal_ranges(alloc), meta_information(alloc), timing_events(alloc) {}
    explicit DataSet(T defaultValue, const allocator_type& alloc = {}) : DataSet(alloc) { default_value = defaultValue; }
    DataSet(const DataSet& other, const allocator_type& alloc) : DataSet(alloc) { *this = other; }
    DataSet(DataSet&& other, const allocator_type& alloc) : DataSet(alloc) { *this = std::move(other); }
    DataSet(const DataSet&)            = default;
    DataSet(DataSet&&) noexcept        = default;
    DataSet& operator=(const DataSet&) = default;
    DataSet& operator=(DataSet&&)      = default;

    [[nodiscard]] allocator_type get_allocator() const noexcept { return signal_values.get_allocator(); }

    [[nodiscard]] std::size_t nDimensions() const noexcept { return extents.size(); }

    [[nodiscard]] std::size_t        axisCount() const noexcept { return axis_names.size(); }
    [[nodiscard]] std::pmr::string&  axisName(std::size_t axisIdx = 0UZ) { return axis_names[_axCheck(axisIdx)]; }
    [[nodiscard]] std::string_view   axisName(std::size_t axisIdx = 0UZ) const { return axis_names[_axCheck(axisIdx)]; }
    [[nodiscard]] std::pmr::string&  axisUnit(std::size_t axisIdx = 0UZ) { return axis_units[_axCheck(axisIdx)]; }
    [[nodiscard]] std::string_view   axisUnit(std::size_t axisIdx = 0UZ) const { return axis_units[_axCheck(axisIdx)]; }
    [[nodiscard]] std::span<T>       axisValues(std::size_t axisIdx = 0UZ) { return axis_values[_axCheck(axisIdx)]; }
    [[nodiscard]] std::span<const T> axisValues(std::size_t axisIdx = 0UZ) const { return axis_values[_axCheck(axisIdx)]; }

    [[nodiscard]] constexpr std::size_t size() const noexcept { return signal_names.size(); }
    [[nodiscard]] std::pmr::string&     signalName(std::size_t signalIdx = 0UZ) { return signal_names[_idxCheck(signalIdx)]; }
    [[nodiscard]] std::string_view      signalName(std::size_t signalIdx = 0UZ) const { return signal_names[_idxCheck(signalIdx)]; }
    [[nodiscard]] std::pmr::string&     signalQuantity(std::size_t signalIdx = 0UZ) { return signal_quantities[_idxCheck(signalIdx)]; }
    [[nodiscard]] std::string_view      signalQuantity(std::size_t signalIdx = 0UZ) const { return signal_quantities[_idxCheck(signalIdx)]; }
    [[nodiscard]] std::pmr::string&     signalUnit(std::size_t signalIdx = 0UZ) { return signal_units[_idxCheck(signalIdx)]; }
    [[nodiscard]] std::string_view      signalUnit(std::size_t signalIdx = 0UZ) const { return signal_units[_idxCheck(signalIdx)]; }
    [[nodiscard]] std::span<T>          signalValues(std::size_t signalIdx = 0UZ) { return {std::next(signal_values.data(), _idxCheckS(signalIdx) * _valsPerSigS()), _valsPerSig()}; }
    [[nodiscard]] std::span<const T>    signalValues(std::size_t signalIdx = 0UZ) const { return {std::next(signal_values.data(), _idxCheckS(signalIdx) * _valsPerSigS()), _valsPerSig()}; }
    [[nodiscard]] Range<T>&             signalRange(std::size_t signalIdx = 0UZ) { return signal_ranges[_idxCheck(signalIdx)]; }
    [[nodiscard]] const Range<T>&       signalRange(std::size_t signalIdx = 0UZ) const { return signal_ranges[_idxCheck(signalIdx)]; }

    [[nodiscard]] pmt_map&                     metaInformation(std::size_t signalIdx = 0UZ) { return meta_information[_idxCheck(signalIdx)]; }
    [[nodiscard]] const pmt_map&               metaInformation(std::size_t signalIdx = 0UZ) const { return meta_information[_idxCheck(signalIdx)]; }
    [[nodiscard]] std::span<idx_pmt_map>       timingEvents(std::size_t signalIdx = 0UZ) { return timing_events[_idxCheck(signalIdx)]; }
    [[nodiscard]] std::span<const idx_pmt_map> timingEvents(std::size_t signalIdx = 0UZ) const { return timing_events[_idxCheck(signalIdx)]; }

    // Clearable conformance.
    void clear() noexcept {
        timestamp = 0;
        axis_names.clear();
        axis_units.clear();
        axis_values.clear();
        extents.clear();
        layout = tensor_layout_type{};
        signal_names.clear();
        signal_quantities.clear();
        signal_units.clear();
        signal_values.clear();
        signal_ranges.clear();
        for (auto& m : meta_information) {
            m.clear();
        }
        meta_information.clear();
        timing_events.clear();
    }

    void shrink_to_fit() {
        axis_names.shrink_to_fit();
        axis_units.shrink_to_fit();
        for (auto& v : axis_values) {
            v.shrink_to_fit();
        }
        axis_values.shrink_to_fit();
        extents.shrink_to_fit();
        signal_names.shrink_to_fit();
        signal_quantities.shrink_to_fit();
        signal_units.shrink_to_fit();
        signal_values.shrink_to_fit();
        signal_ranges.shrink_to_fit();
        for (auto& m : meta_information) {
            m.shrink_to_fit();
        }
        meta_information.shrink_to_fit();
        for (auto& ev : timing_events) {
            ev.shrink_to_fit();
        }
        timing_events.shrink_to_fit();
    }

private:
    [[nodiscard]] std::size_t _axCheck(std::size_t i, std::source_location loc = std::source_location::current()) const {
        if (i >= axis_names.size()) {
            gr::log::fatal(gr::log::runtime("{} axis out of range: i={} >= axis_name [0, {}]", loc), loc.function_name(), i, axis_names.size());
        }
        if (i >= axis_values.size()) {
            gr::log::fatal(gr::log::runtime("{} axis out of range: i={} >= axis_values [0, {}]", loc), loc.function_name(), i, axis_values.size());
        }
        return i;
    }

    [[nodiscard]] std::size_t _idxCheck(std::size_t i, std::source_location location = std::source_location::current()) const {
        if (i >= size()) {
            gr::log::fatal(gr::log::runtime("{} out of range: i={} >= [0, {}]", location), location.function_name(), i, size());
        }
        return i;
    }

    [[nodiscard]] std::ptrdiff_t _idxCheckS(std::size_t i, std::source_location location = std::source_location::current()) const {
        if (i >= size()) {
            gr::log::fatal(gr::log::runtime("{} out of range: i={} >= [0, {}]", location), location.function_name(), i, size());
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
    using value_type     = T;
    using allocator_type = std::pmr::polymorphic_allocator<>;
    using pmt_map        = property_map;
    T default_value      = T(); // default value for padding, ZOH etc.

    std::int64_t              timestamp = 0;   // UTC timestamp [ns]
    std::pmr::vector<T>       signal_values{}; // size = \PI_i extents[i
    std::pmr::vector<pmt_map> meta_information{};

    GR_MAKE_REFLECTABLE(Packet, default_value, timestamp, signal_values, meta_information);

    Packet() = default;
    explicit Packet(const allocator_type& alloc) : signal_values(alloc), meta_information(alloc) {}
    explicit Packet(T defaultValue, const allocator_type& alloc = {}) : Packet(alloc) { default_value = defaultValue; }
    Packet(const Packet& other, const allocator_type& alloc) : Packet(alloc) { *this = other; }
    Packet(Packet&& other, const allocator_type& alloc) : Packet(alloc) { *this = std::move(other); }
    Packet(const Packet&)            = default;
    Packet(Packet&&) noexcept        = default;
    Packet& operator=(const Packet&) = default;
    Packet& operator=(Packet&&)      = default;

    [[nodiscard]] allocator_type get_allocator() const noexcept { return signal_values.get_allocator(); }
};

static_assert(PacketLike<Packet<std::byte>>, "Packet<std::byte> concept conformity");
static_assert(PacketLike<Packet<float>>, "Packet<std::byte> concept conformity");
static_assert(PacketLike<Packet<double>>, "Packet<std::byte> concept conformity");

} // namespace gr

#endif // GNURADIO_DATASET_HPP
