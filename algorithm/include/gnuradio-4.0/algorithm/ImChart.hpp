#ifndef GNURADIO_IMCHART_HPP
#define GNURADIO_IMCHART_HPP

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <numeric>
#include <ranges>
#include <source_location>
#ifdef _WIN32
#include <windows.h>
#else
#include <sys/ioctl.h>
#endif
#include <vector>

#include <format>
#ifdef __GNUC__
#pragma GCC diagnostic push // ignore warning of external libraries that from this lib-context we do not have any control over
#ifndef __clang__
#pragma GCC diagnostic ignored "-Wuseless-cast"
#endif
#pragma GCC diagnostic ignored "-Wsign-conversion"
#endif
#include <magic_enum.hpp>
#include <magic_enum_utility.hpp>
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

#include "gnuradio-4.0/meta/reflection.hpp"
#include "gnuradio-4.0/meta/utils.hpp"
#include <bitset>
#include <iostream>

namespace gr::graphs {

class Color {
public:
    enum class Type : std::uint8_t { Default, Blue, Red, Green, Yellow, Magenta, Cyan, LightBlue, LightRed, LightGreen, LightYellow, LightMagenta, LightCyan, White, LightGray, DarkGray, Black };

    [[nodiscard]] constexpr static const char* get(Type colour) noexcept {
        for (const auto& [colType, colStr] : Colors) {
            if (colType == colour) {
                return colStr;
            }
        }
        return Colors[0].second; // fallback, should not happen if Colors array is correctly defined.
    }

    [[nodiscard]] constexpr static const char* get(std::size_t index) noexcept { return Colors[index % Colors.size()].second; }

    [[nodiscard]] constexpr static uint8_t getIndex(Type colour) noexcept { return static_cast<uint8_t>(colour); }

    [[nodiscard]] constexpr static Type next(Type colour) noexcept { return magic_enum::enum_next_value_circular(colour); }

    [[nodiscard]] constexpr static Type prev(Type colour) noexcept { return magic_enum::enum_prev_value_circular(colour); }

private:
    constexpr static std::array<std::pair<Type, const char*>, 17UZ> Colors = {std::make_pair(Type::Default, "\x1B[39m"),                                                                                                                                                                   //
        std::make_pair(Type::Blue, "\x1B[34m"), std::make_pair(Type::Red, "\x1B[31m"), std::make_pair(Type::Green, "\x1B[32m"), std::make_pair(Type::Yellow, "\x1B[33m"), std::make_pair(Type::Magenta, "\x1B[35m"), std::make_pair(Type::Cyan, "\x1B[36m"),                               //
        std::make_pair(Type::LightBlue, "\x1B[94m"), std::make_pair(Type::LightRed, "\x1B[91m"), std::make_pair(Type::LightGreen, "\x1B[92m"), std::make_pair(Type::LightYellow, "\x1B[93m"), std::make_pair(Type::LightMagenta, "\x1B[95m"), std::make_pair(Type::LightCyan, "\x1B[96m"), //
        std::make_pair(Type::White, "\x1B[97m"), std::make_pair(Type::LightGray, "\x1B[37m"), std::make_pair(Type::DarkGray, "\x1B[90m"), std::make_pair(Type::Black, "\x1B[30m")};
};

struct LinearAxisTransform {
    template<std::floating_point T>
    [[nodiscard]] static constexpr std::size_t toScreen(T value, T axisMin, T axisMax, std::size_t screenOffset, std::size_t screenSize) {
        return screenOffset + static_cast<std::size_t>((value - axisMin) / (axisMax - axisMin) * static_cast<T>(screenSize - screenOffset - 1UZ));
    }

    template<std::floating_point T>
    [[nodiscard]] static constexpr T fromScreen(std::size_t screenCoordinate, T axisMin, T axisMax, std::size_t screenOffset, std::size_t screenSize) {
        return axisMin + static_cast<T>(screenCoordinate - screenOffset) / static_cast<T>(screenSize - screenOffset - 1UZ) * (axisMax - axisMin);
    }
};

struct LogAxisTransform {
    template<typename T>
    [[nodiscard]] static constexpr std::size_t toScreen(T value, T axisMin, T axisMax, std::size_t screenOffset, std::size_t screenSize) {
        if (value <= 0 || axisMin <= 0 || axisMax <= axisMin) {
            throw std::invalid_argument(std::format("{} not defined for non-positive value {} in [{}, {}].", gr::meta::type_name<LogAxisTransform>(), value, axisMin, axisMax));
        }

        const T log_min    = std::log10(axisMin);
        const T proportion = (std::log10(value) - log_min) / (std::log10(axisMax) - log_min);                         // interpolation in log space
        return screenOffset + static_cast<std::size_t>(proportion * static_cast<T>(screenSize - screenOffset - 1UZ)); // convert into screen space
    }

    template<std::floating_point T>
    [[nodiscard]] static constexpr T fromScreen(std::size_t screenCoordinate, T axisMin, T axisMax, std::size_t screenOffset, std::size_t screenSize) {
        if (axisMin <= 0UZ || axisMax <= axisMin) {
            throw std::invalid_argument(std::format("{} not defined for non-positive ranges [{}, {}].", gr::meta::type_name<LogAxisTransform>(), axisMin, axisMax));
        }

        const T proportion = static_cast<T>(screenCoordinate - screenOffset) / (static_cast<T>(screenSize - screenOffset - 1UZ)); // convert screen coordinates back to a proportion of the axis
        const T log_min    = std::log10(axisMin);
        return std::pow(static_cast<T>(10), log_min + proportion * (std::log10(axisMax) - log_min)); // interpolate in the log space and 10^x
    }
};

enum struct ResetChartView { RESET, KEEP };

enum class Style { Braille, Bars, Marker };

namespace detail {
inline std::vector<std::size_t> optimalTickScreenPositions(std::size_t axisWidth, std::size_t minGapSize = 1) {
    constexpr std::array preferredDivisors{10UZ, 8UZ, 5UZ, 4UZ, 3UZ, 2UZ};
    std::size_t          reducedAxisWidth = axisWidth - 1; // because we always require & add the '0'

    // checks if preferred divisor evenly divides the 'axisWidth - 1'
    auto validDivisorIt = std::ranges::find_if(preferredDivisors, [&](std::size_t divisor) { return reducedAxisWidth % divisor == 0 && (reducedAxisWidth / divisor) > minGapSize; });

    // determine the segment size.
    std::size_t segmentSize = validDivisorIt != preferredDivisors.end() ? (reducedAxisWidth < 10 ? *validDivisorIt : (reducedAxisWidth / *validDivisorIt)) : reducedAxisWidth; // default -> [0, reducedAxisWidth]

    auto tickRange = std::views::iota(0UZ, axisWidth) | std::views::filter([=](auto i) { return i % segmentSize == 0; });
    return {tickRange.begin(), tickRange.end()};
}

} // namespace detail

inline void resetView() { std::puts("\033[2J\033[H"); }

/**
 * @brief compact class for ASCII charting in terminal environments, supporting custom dimensions and styles.
 *
 * ImChart enables the creation of various graph styles, including bar, braille, and point markers, directly within the terminal.
 * Unicode braille characters are used for compact high-density visualisation, and supports dynamic scaling and some customisation.
 * The primary use-case-focus is for basic visual debugging for library developer for FIR/IIR filter generation, Bode plots, and
 * other basic applications w/o having to spin-up a more complete graphical UI-based solution.
 *
 * Usage:
 *
 * @subsection basic_example Basic Example
 *   ImChart<80, 30> chart;  // use 80 columns x 30 rows
 * @code
 *   ImChart<80, 30> chart;  // use 80 columns x 30 rows
 *   std::vector<double> xValues, sineValuesY, cosineValuesY; // your data
 *   // ... initialise data
 *   chart.draw(xValues, sineValuesY, "sine-like"); // draws a sine-like line
 *   chart.draw(xValues, cosineValuesY, "cosine-like"); // draws a cosine-like line
 *   chart.draw();
 * @endcode
 *
 * @subsection bar_example Chart with bar elements, often used for histograms or similar data visualizations.
 * @code
 *   auto chart ImChart<80, 30>({ {0.0, 100.0}, {0.0, 5.0} }); // setting chart boundaries
 *   chart.draw_border = true; // optional border around chart
 *   // ... initialise data
 *   chart.draw<Style::Bars>(xValues, barData, "bar-data");
 *   chart.draw();
 * @endcode
 *
 * @subsection marker_example Chart with with individual point markers.
 * @code
 *   auto chart ImChart<80, 30>(); // setting w/o boundaries -> auto-determined by first dataset
 *   // ... initialise data
 *   chart.draw<Style::Marker>(xValues, pointData, "point-data"); // draws points
 *   chart.draw();
 * @endcode
 *
 * @subsection log_example Log-Axis Chart
 * @code
 *   auto = chartImChart<120, 16, LogAxisTransform>({ {0.1, 10000.0}, {-100, 0} }); // x-log axis
 *   // ... initialise data
 *   chart.draw(xValues, responseData, "response-data"); // draws the data
 *   chart.draw(); // render the chart
 * @endcode
 *
 * @subsection axis_transform Custom Axis Transform
 * You can replace the LinearAxisTransform (default) or LogAxisTransform with your custom transform provided:
 * @code
 *   struct MyAxisTransform {
 *     template<std::floating_point T>
 *     static std::size_t toScreen(T value, T axisMin, T axisMax, std::size_t screenOffset, std::size_t screenSize);
 *     template<std::floating_point T>
 *     static T fromScreen(std::size_t screenCoordinate, T axisMin, T axisMax, std::size_t screenOffset, std::size_t screenSize);
 *   };
 *   double value = 42.0; // or any other value to be transformed
 *   auto screenCoordinate = LinearAxisTransform::toScreen(value, 10., 100., 5, 65);
 *   auto reconstructedValue = LinearAxisTransform::fromScreen(screenCoordinate, 10., 100., 5, 65); // test: reconstructedValue == ~value
 * @endcode
 *
 */
template<std::size_t screenWidth, std::size_t screenHeight, typename horAxisTransform = LinearAxisTransform, typename verAxisTransform = LinearAxisTransform>
struct ImChart {
    std::conditional_t<screenWidth != std::dynamic_extent, const std::size_t, std::size_t>  _screen_width{screenWidth};
    std::conditional_t<screenHeight != std::dynamic_extent, const std::size_t, std::size_t> _screen_height{screenHeight};

    //
    constexpr static std::size_t                                              kCellWidth{2U};
    constexpr static std::size_t                                              kCellHeight{4U};
    constexpr static std::array<const char*, 256>                             kBrailleCharacter{"⠀", "⠁", "⠂", "⠃", "⠄", "⠅", "⠆", "⠇", "⠈", "⠉", "⠊", "⠋", "⠌", "⠍", "⠎", "⠏", "⠐", "⠑", "⠒", "⠓", "⠔", "⠕", "⠖", "⠗", "⠘", "⠙", "⠚", "⠛", "⠜", "⠝", "⠞", "⠟", "⠠", "⠡", "⠢", "⠣", "⠤", "⠥", "⠦", "⠧", "⠨", "⠩", "⠪", "⠫", "⠬", "⠭", "⠮", "⠯", "⠰", "⠱", "⠲", "⠳", "⠴", "⠵", "⠶", "⠷", "⠸", "⠹", "⠺", "⠻", "⠼", "⠽", "⠾", "⠿", "⡀", "⡁", "⡂", "⡃", "⡄", "⡅", "⡆", "⡇", "⡈", "⡉", "⡊", "⡋", "⡌", "⡍", "⡎", "⡏", "⡐", "⡑", "⡒", "⡓", "⡔", "⡕", "⡖", "⡗", "⡘", "⡙", "⡚", "⡛", "⡜", "⡝", "⡞", "⡟", "⡠", "⡡", "⡢", "⡣", "⡤", "⡥", "⡦", "⡧", "⡨", "⡩", "⡪", "⡫", "⡬", "⡭", "⡮", "⡯", "⡰", "⡱", "⡲", "⡳", "⡴", "⡵", "⡶", "⡷", "⡸", "⡹", "⡺", "⡻", "⡼", "⡽", "⡾", "⡿", "⢀", "⢁", "⢂", "⢃", "⢄", "⢅", "⢆", "⢇", "⢈", "⢉", "⢊", "⢋", "⢌", "⢍", "⢎", "⢏", "⢐", "⢑", "⢒", "⢓", "⢔", "⢕", "⢖", "⢗", "⢘", "⢙", "⢚", "⢛", "⢜", "⢝", "⢞", "⢟", "⢠", "⢡", "⢢", "⢣", "⢤", "⢥", "⢦", "⢧", "⢨", "⢩", "⢪", "⢫", "⢬", "⢭", "⢮", "⢯", "⢰", "⢱", "⢲", "⢳", "⢴", "⢵", "⢶", "⢷", "⢸", "⢹", "⢺", "⢻", "⢼", "⢽", "⢾", "⢿", "⣀", "⣁", "⣂", "⣃", "⣄", "⣅", "⣆", "⣇", "⣈", "⣉", "⣊", "⣋", "⣌", "⣍", "⣎", "⣏", "⣐", "⣑", "⣒", "⣓", "⣔", "⣕", "⣖", "⣗", "⣘", "⣙", "⣚", "⣛", "⣜", "⣝", "⣞", "⣟", "⣠", "⣡", "⣢", "⣣", "⣤", "⣥", "⣦", "⣧", "⣨", "⣩", "⣪", "⣫", "⣬", "⣭", "⣮", "⣯", "⣰", "⣱", "⣲", "⣳", "⣴", "⣵", "⣶", "⣷", "⣸", "⣹", "⣺", "⣻", "⣼", "⣽", "⣾", "⣿"};
    constexpr static std::array<std::array<uint8_t, kCellHeight>, kCellWidth> kBrailleDotMap{{{0x1, 0x2, 0x4, 0x40}, {0x8, 0x10, 0x20, 0x80}}};
    // bar definitions
    constexpr static std::array<const char*, 9> kBars{" ", "▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"};
    static_assert(static_cast<std::size_t>(std::ranges::distance(kBars)) >= kCellWidth * kCellHeight, "bar definitions must be >= kCellWidth * kCellHeight");
    constexpr static std::array<const char*, 9> kMarker{"X", "O", "★", "+", "❖", "◎", "○", "■", "□"};

    constexpr static Color::Type          kFirstColor{Color::Type::Blue}; // we like blue
    std::vector<std::vector<std::string>> _screen;
    // _brailleArray is the 2x4 (kCellWidth k CellHeight) oversampled data array that is used
    // to in turn compute the required braille and bar characters that are inserted into the _screen array
    // first byte : stores values
    // second byte: stores bitmask which data set (max: 8) is involved in the screen character
    std::vector<std::vector<uint16_t>> _brailleArray;

    Color::Type              _lastColor = kFirstColor;
    std::size_t              _n_datasets{0UZ};
    std::vector<std::string> _datasets{};
    std::source_location     _location;

public:
    std::string axis_name_x = "x-axis []";
    std::string axis_name_y = "y-axis []";
    bool        draw_border = false;
    double      axis_min_x{0.0};
    double      axis_max_x{0.0};
    double      axis_min_y{0.0};
    double      axis_max_y{0.0};
    std::size_t n_ticks_x = std::min(10UZ, screenWidth / 2UZ);
    std::size_t n_ticks_y = std::min(10UZ, screenHeight / 2UZ);

    constexpr ImChart(std::size_t screenWidth_ = screenWidth, std::size_t screenHeight_ = screenHeight, const std::source_location location = std::source_location::current()) noexcept : _screen_width(screenWidth_), _screen_height(screenHeight_), _screen(_screen_height, std::vector<std::string>(_screen_width, " ")), _brailleArray(_screen_width * kCellWidth, std::vector<uint16_t>(_screen_height * kCellHeight, 0UZ)), _location(location) {}

    explicit ImChart(const std::tuple<std::pair<double, double>, std::pair<double, double>>& init, std::size_t screenWidth_ = screenWidth, std::size_t screenHeight_ = screenHeight) : ImChart(screenWidth_, screenHeight_) {
        const auto& [xBounds, yBounds] = init;
        axis_min_x                     = xBounds.first;
        axis_max_x                     = xBounds.second;
        axis_min_y                     = yBounds.first;
        axis_max_y                     = yBounds.second;
    }

    template<Style style = Style::Braille, std::ranges::input_range TContainer1, std::ranges::input_range TContainer2>
    void draw(const TContainer1& xValues, const TContainer2& yValues, std::string_view datasetName = {}) {
        static_assert(std::is_same_v<std::ranges::range_value_t<TContainer1>, std::ranges::range_value_t<TContainer2>>, "x- and y- range must have same value_type");
        using ValueType = typename std::ranges::range_value_t<TContainer1>;
        static_assert(std::is_arithmetic_v<ValueType>, "collection's value_type must be a arithmetic type!");
        if (xValues.size() != yValues.size() || xValues.empty()) {
            return;
        }
        _n_datasets++;
        if (axis_min_x == axis_max_x) {
            const auto [minX, maxX] = std::ranges::minmax_element(xValues);
            axis_min_x              = static_cast<double>(*minX);
            axis_max_x              = static_cast<double>(*maxX);
            if (axis_min_x == axis_max_x) {
                axis_max_x += 1.0; // safe fall back for x-range
            }
        }
        if (axis_min_y == axis_max_y) {
            constexpr double padding = 0.05;
            const auto [minY, maxY]  = std::ranges::minmax_element(yValues);
            axis_min_y               = static_cast<double>(*minY) - padding * static_cast<double>(*maxY - *minY);
            axis_max_y               = static_cast<double>(*maxY) + padding * static_cast<double>(*maxY - *minY);
            if (axis_min_y == axis_max_y) {
                axis_max_y += 1.0; // safe fall back for y-range
            }
        }

#if defined(__EMSCRIPTEN__) || defined(__clang__) // TODO: remove once the CI has a new clang/libc++
        if (std::ranges::find(_datasets.cbegin(), _datasets.cend(), datasetName) == _datasets.end()) {
            _datasets.emplace_back(datasetName);
        }
#else
        if (!std::ranges::contains(_datasets, datasetName)) {
            _datasets.emplace_back(datasetName);
        }
#endif

        // clear braille array
        std::ranges::for_each(_brailleArray, [](auto& line) { std::ranges::fill(line, 0UZ); });
        auto updateBrailleArray = [this](const uint16_t oldValue) -> uint16_t {
            uint16_t increment = ((oldValue & 0xFF) + 1UZ) & 0xFF;     // Increment and mask to first byte
            uint16_t colorBit  = ((1UZ << _n_datasets) << 8) & 0xFF00; // Shift to the second byte
            return increment | colorBit;
        };

        const std::size_t horAxisPosY = getHorizontalAxisPositionY() * kCellHeight;
        const auto        arrayWidth  = _screen_width * kCellWidth;
        const auto        arrayHeight = _screen_height * kCellHeight;
        const std::size_t horOffset   = (axis_min_x == 0.0 || std::is_same_v<horAxisTransform, LogAxisTransform>) ? getVerticalAxisPositionX() * kCellWidth : 0UZ;
#if defined(__EMSCRIPTEN__) || defined(__clang__) // TODO: remove once the CI has a new clang/libc++
        auto xIt = xValues.begin();
        auto yIt = yValues.begin();

        for (; xIt != xValues.end() && yIt != yValues.end(); ++xIt, ++yIt) {
            auto x = *xIt;
            auto y = *yIt;
#else
        for (const auto& [x, y] : std::ranges::views::zip(xValues, yValues)) {
#endif
            if (static_cast<double>(x) < axis_min_x || static_cast<double>(x) >= axis_max_x || static_cast<double>(y) < axis_min_y || static_cast<double>(y) >= axis_max_y) {
                continue;
            }

            const auto brailleColIndex = horAxisTransform::toScreen(x, ValueType(axis_min_x), ValueType(axis_max_x), horOffset, arrayWidth);
            const auto brailleRowIndex = arrayHeight - verAxisTransform::toScreen(y, ValueType(axis_min_y), ValueType(axis_max_y), 0UZ, arrayHeight);
            if (brailleRowIndex >= (arrayHeight - 1UZ) || brailleColIndex >= arrayWidth || (style == Style::Bars && brailleRowIndex >= horAxisPosY)) {
                continue;
            }

            _brailleArray[brailleColIndex][brailleRowIndex] = updateBrailleArray(_brailleArray[brailleColIndex][brailleRowIndex]);

            if constexpr (style == Style::Bars) {
                // fill towards x-axis
                std::size_t start = std::min(brailleRowIndex, horAxisPosY);
                std::size_t end   = std::max(brailleRowIndex, horAxisPosY);
                for (std::size_t fillPos = start; fillPos <= end; ++fillPos) {
                    _brailleArray[brailleColIndex][fillPos] = updateBrailleArray(_brailleArray[brailleColIndex][fillPos]);
                }
            }
        }
        for (std::size_t bRowIdx = 0UZ; bRowIdx < arrayHeight; bRowIdx += kCellHeight) {
            for (std::size_t bColIdx = 0UZ; bColIdx < arrayWidth; bColIdx += kCellWidth) {
                // integrate over 4x2 braille pattern
                uint8_t  dot         = 0UZ;
                uint16_t datasetMask = 0U;
                for (std::size_t k = 0UZ; k < kCellWidth; ++k) {
                    for (std::size_t l = 0UZ; l < kCellHeight; ++l) {
                        const uint16_t brailleValue   = _brailleArray[bColIdx + k][bRowIdx + l]; // normally 1, >1 if there is an overlap
                        uint16_t       firstByteValue = brailleValue & 0xFF;
                        if (firstByteValue) {
                            datasetMask |= (brailleValue >> 8) & 0xFFFFFFFF;
                            if constexpr (style == Style::Bars) {
                                dot++; // only increase by one so that sum remains <= 8 (<->  kCellHeight(4) * kCellWidth(2))
                            } else {
                                dot += kBrailleDotMap[k][l];
                            }
                        }
                    }
                }

                const bool datasetInvolved = (datasetMask & (1UZ << _n_datasets)) > 0;
                const bool overlapDetected = std::popcount(datasetMask) > 1;
                if (dot == 0 && !overlapDetected && (_screen[bRowIdx / kCellHeight][bColIdx / kCellWidth] == " ")) {
                    continue;
                }
                if (!datasetInvolved) {
                    continue;
                }
                const auto colourStr = Color::get(_lastColor);
                auto&      screen    = _screen[bRowIdx / kCellHeight][bColIdx / kCellWidth];
                screen.clear();
                screen += colourStr;

                switch (style) {
                case Style::Bars: screen += kBars[dot]; break;
                case Style::Marker: screen += kMarker[_n_datasets - 1]; break;
                case Style::Braille:
                default: screen += kBrailleCharacter[dot]; break;
                }
                screen += Color::get(Color::Type::Default);
            }
        }
        _lastColor = Color::next(_lastColor);
    }

    void draw(std::source_location caller = std::source_location::current()) {
        _location = caller;
        drawBorder();
        drawAxes();
        drawLegend();

        printSourceLocation();
        printScreen();
    }

    void clearScreen() noexcept {
        std::ranges::for_each(_screen, [](auto& line) { std::ranges::fill(line, " "); });
        std::ranges::for_each(_brailleArray, [](auto& line) { std::ranges::fill(line, 0UZ); });
        _lastColor  = kFirstColor;
        _n_datasets = 0UZ;
    }

    void reset() const noexcept { std::puts("\033[0;0H"); }

    void printScreen() const noexcept {
        for (const auto& row : _screen) {
            for (const auto& cell : row) {
                std::fwrite(cell.data(), 1, cell.size(), stdout);
            }
            std::fputs("\n", stdout);
        }
        // Reset terminal colour after printing
        std::fputs(Color::get(Color::Type::Default), stdout);
    }

    void printSourceLocation() {
        const std::size_t maxLength = _screen_width / 4UZ;
        std::string       fullPath(_location.file_name());

        if (fullPath.size() > maxLength) { // check if path needs to be truncated
            std::size_t cutPosition = fullPath.find_last_of('/', fullPath.size() - maxLength);
            fullPath                = (cutPosition != std::string::npos) ? "[..]" + fullPath.substr(cutPosition) : std::move(fullPath);
        }
        std::string srcLocation = std::format("{}:{}", fullPath, _location.line());

        // calculate starting position, clamping to screen width
        std::size_t startX = std::max(0UZ, _screen_width - srcLocation.size() - 1UZ);
        std::size_t y      = _screen_height - 1; // position for the last line

        for (std::size_t i = 0; i < srcLocation.size() && startX + i < _screen_width; ++i) {
            _screen[y][startX + i] = srcLocation[i];
        }
    }

    [[nodiscard]] constexpr std::size_t getHorizontalAxisPositionY() const noexcept {
        const double relative_position = (0.0 - axis_min_y) / (axis_max_y - axis_min_y);
        const auto   position          = static_cast<std::size_t>((1.0 - relative_position) * static_cast<double>(_screen_height));
        return std::clamp(position, 0UZ, _screen_height - 3UZ);
    }

    [[nodiscard]] constexpr std::size_t getVerticalAxisPositionX() const noexcept {
        auto y_axis_x = std::is_same_v<horAxisTransform, LogAxisTransform> ? 0UZ : static_cast<std::size_t>((std::max(0. - axis_min_x, 0.) / (axis_max_x - axis_min_x)) * static_cast<double>(_screen_width - 1UZ));
        // adjust for axis labels
        std::size_t y_label_width = std::max(std::format("{:G}", axis_min_y).size(), std::format("{:G}", axis_max_y).size());
        return std::clamp(y_axis_x, y_label_width + 3, _screen_width); // Ensure axis positions are within screen bounds
    }

    void drawAxes() {
        const std::size_t horAxisPosY = getHorizontalAxisPositionY();
        const std::size_t verAxisPosX = getVerticalAxisPositionX();
        const std::size_t horOffset   = (axis_min_x == 0.0 || std::is_same_v<horAxisTransform, LogAxisTransform>) ? verAxisPosX : 0UZ;

        // drawing the axes
        for (std::size_t i = horOffset + 1; i < _screen_width - 1; ++i) {
            _screen[horAxisPosY][i] = "─";
        }

        for (std::size_t i = 1; i < _screen_height - 1; ++i) {
            _screen[i][verAxisPosX] = "│";
        }

        // x-axis labels and ticks
        const std::size_t maxHorLabelWidth = std::max(std::format("{:+G}", axis_min_x).size(), std::format("{:+G}", axis_max_x).size());
        for (const auto& relTickScreenPos : detail::optimalTickScreenPositions(_screen_width - horOffset, 1UZ)) {
            const std::size_t tickPos   = horOffset + relTickScreenPos;
            const double      tickValue = horAxisTransform::fromScreen(tickPos, axis_min_x, axis_max_x, horOffset, _screen_width);
            if ((axis_min_x == 0 && relTickScreenPos == 0) || tickPos > _screen_width) {
                continue; // skip first '0' that would collide with the vertical axis
            }
            _screen[horAxisPosY][tickPos] = relTickScreenPos == 0 ? "┌" : ((tickPos + 1) >= _screen_width) ? "┐" : "┬"; // NOSONAR

            const std::string rawLabel = axis_min_x < 0 ? std::format("{:+G}", tickValue) : std::format("{:G}", tickValue);
            const std::string label    = std::format("{:.{}}", rawLabel, maxHorLabelWidth);

            // Calculate the start and end positions for the label to be centred around tickPos and clamp to ensure they're within screen bounds
            const std::size_t start_pos = std::clamp((tickPos >= label.size() / 2) ? tickPos - label.size() / 2 : 0, 0UZ, _screen_width - label.size());
            const std::size_t end_pos   = std::clamp(start_pos + label.size(), label.size(), _screen_width);

            for (std::size_t i = 0; start_pos + i < end_pos && i < label.size(); i++) {
                _screen[horAxisPosY + 1][start_pos + i] = label[i];
            }
        }

        // y-axis labels and ticks
        for (const auto& relTickScreenPos : detail::optimalTickScreenPositions(_screen_height)) {
            const std::size_t tickPos   = _screen_height - 1UZ - relTickScreenPos;
            const double      tickValue = verAxisTransform::fromScreen(relTickScreenPos, axis_min_y, axis_max_y, 0UZ, _screen_height);
            if (/*(axis_min_y == 0 && relTickScreenPos == 0) ||*/ tickPos > _screen_height) {
                continue; // skip first '0' that would collide with the vertical axis
            }
            _screen[tickPos][verAxisPosX] = relTickScreenPos == 0 ? "┘" : (tickPos < _screen_height) ? "┤" : (axis_max_y == 0) ? "┬" : "┐"; // NOSONAR

            const std::string rawLabel = axis_min_y < 0 ? std::format("{:+G}", tickValue) : std::format("{:G}", tickValue);
            const std::string label    = std::format("{:.{}}", rawLabel, verAxisPosX - 2UZ);

            // Calculate the starting position for the label ensuring it's within bounds.
            const std::size_t label_start_pos = (verAxisPosX > label.size() + 1UZ) ? verAxisPosX - label.size() - 1UZ : 0UZ;
            for (std::size_t i = 0UZ; i < label.size(); i++) {
                _screen[tickPos][label_start_pos + i] = label[i];
            }
        }

        // drawing y-axis label to right of the y-axis
        for (std::size_t i = 0; i < axis_name_y.size() && i + verAxisPosX + 1 < _screen_width; i++) {
            _screen[0][verAxisPosX + 1 + i] = axis_name_y[i];
        }

        // drawing x-axis label right-aligned and below the last tick value
        const std::size_t label_x_start_pos = (_screen_width >= axis_name_x.size()) ? _screen_width - axis_name_x.size() : 0;
        for (std::size_t i = 0; label_x_start_pos + i < _screen_width && i < axis_name_x.size(); i++) {
            _screen[horAxisPosY + 2][label_x_start_pos + i] = axis_name_x[i];
        }

        // axes intersection
        if (horAxisPosY > 0 && horAxisPosY < _screen_width && verAxisPosX > 0 && verAxisPosX < _screen_height) {
            _screen[horAxisPosY][verAxisPosX] = axis_min_x == 0.0 ? "├" : "┼";
        }

        // border intersections
        if (draw_border) {
            _screen[0][verAxisPosX]                  = "┬";
            _screen[_screen_height - 1][verAxisPosX] = "┴";
            _screen[horAxisPosY][0]                  = "├";
            _screen[horAxisPosY][_screen_width - 1]  = "┤";
        }
    }

    void drawLegend() {
        const std::size_t cursorY = _screen_height - 1LU; // legend position on last row
        std::size_t       cursorX = getVerticalAxisPositionX() + 2LU;

        auto colour = kFirstColor;
        for (const auto& datasetName : _datasets) {
            if (datasetName.empty()) {
                continue;
            }

            _screen[cursorY][cursorX++] = std::string(Color::get(colour)) + "█" + Color::get(Color::Type::Default);
            colour                      = Color::next(colour);

            // ad a separator (': ')
            _screen[cursorY][cursorX++].assign(":");
            _screen[cursorY][cursorX++].assign(" ");

            // add the dataset name
            for (std::size_t j = 0; j < datasetName.size() && cursorX + j < _screen_width; j++) {
                _screen[cursorY][cursorX++] = datasetName[j];
            }

            if (cursorX >= _screen_width) { // wrote till the end of the screen
                return;
            }

            _screen[cursorY][cursorX++] = ' '; // add a space separator between dataset names
        }
    }

    void drawBorder() {
        if (!draw_border) {
            return;
        }
        // Top and bottom horizontal edges
        std::ranges::fill(_screen[0], "─");
        std::ranges::fill(_screen[_screen_height - 1], "─");

        // Vertical edges
        for (std::size_t i = 1UZ; i < _screen_height - 1; ++i) {
            _screen[i][0]                 = "│";
            _screen[i][_screen_width - 1] = "│";
        }

        // Corners
        _screen[0][0]                                  = "┌";
        _screen[0][_screen_width - 1]                  = "┐";
        _screen[_screen_height - 1][0]                 = "└";
        _screen[_screen_height - 1][_screen_width - 1] = "┘";
    }
};

} // namespace gr::graphs

#endif // GNURADIO_IMCHART_HPP
