#ifndef GNURADIO_IMGRAPH_HPP
#define GNURADIO_IMGRAPH_HPP

#include "gnuradio-4.0/Graph.hpp"
#include "gnuradio-4.0/Message.hpp"
#include "gnuradio-4.0/Tag.hpp"

#include <gnuradio-4.0/algorithm/ImCanvas.hpp>

#include <gnuradio-4.0/meta/formatter.hpp>
#include <gnuradio-4.0/meta/utils.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <bitset>
#include <cassert>
#include <cmath>
#include <format>
#include <functional>
#include <limits>
#include <map>
#include <numbers>
#include <numeric>
#include <optional>
#include <print>
#include <queue>
#include <ranges>
#include <set>
#include <source_location>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

namespace gr::graph {
using gr::utf8::Direction;
using gr::utf8::Point;
using gr::utf8::Style;
using gr::utf8::color::Colour;

[[nodiscard]] constexpr static const char* utf8_digits(std::size_t idx, bool white = true) {
    constexpr static auto encircledWhite = std::to_array<const char*>({//
        "⓪", "①", "②", "③", "④", "⑤", "⑥", "⑦", "⑧", "⑨",              //
        "⑩", "⑪", "⑫", "⑬", "⑭", "⑮", "⑯", "⑰", "⑱", "⑲",              //
        "⑳", "㉑", "㉒", "㉓", "㉔", "㉕", "㉖", "㉗", "㉘", "㉙",     //
        "㉚", "㉛", "㉜", "㉝", "㉞", "㉟", "㊱", "㊲", "㊳", "㊴",    //
        "㊵", "㊶", "㊷", "㊸", "㊹", "㊺", "㊻", "㊼", "㊽", "㊾", "㊿"});
    constexpr static auto encircledBlack = std::to_array<const char*>({//
        "⓿", "❶", "❷", "❸", "❹", "❺", "❻", "❼", "❽", "❾",              //
        "❿", "⓫", "⓬", "⓭", "⓮", "⓯", "⓰", "⓱", "⓲", "⓳", "⓴"});
    return white ? encircledWhite[idx & encircledWhite.size()] : encircledBlack[idx & encircledBlack.size()];
}

//------------------------------------------------------
// Phase 0: Core data model
//------------------------------------------------------
enum class LayoutPref { HORIZONTAL, VERTICAL, AUTO, UNDEFINED };
enum class Side : std::size_t { Left = 0UZ, Right = 1UZ, Top = 2UZ, Bottom = 3UZ };
enum class EdgeType { Forward, Feedback, Lateral };

Side oppositeSide(Side s) noexcept {
    using enum Side;
    switch (s) {
    case Left: return Right;
    case Right: return Left;
    case Top: return Bottom;
    case Bottom: return Top;
    }
    return s; // unreachable
}

enum class LayoutAlgorithm {
    Sugiyama,            /// Sugiyama, Tagawa, Toda ’81
    EadesLog,            ///
    FruchtermanReingold, ///
    Tutte,               ///
    NormalizedRadial,    // my original implementation (to be kept for reference)
    ManhattanPerAxis     ///
};

struct LayoutPreference {
    LayoutPref      layout                    = LayoutPref::HORIZONTAL;
    LayoutAlgorithm algorithm                 = LayoutAlgorithm::Sugiyama;
    double          kSpring                   = 5.0; // per-axis spring gain (|dx|,|dy| manhattan components)
    double          kCoulombPower             = 2UZ; // 1 => 1/r, 2 => 1/r²
    double          kFruchtermanReingoldScale = 1.0; // global Fruchterman-Reingold scale 'C'
    double          kRepel                    = 1.0; // spacing/overlap repulsion
    std::size_t     minDefaultGap             = 1UZ; // minimum gap if no other constraints apply
    std::size_t     minGap                    = 2UZ; // minimum gap between adjacent blocks
    std::size_t     minMargin                 = 2UZ; // minimum gap to the left or top border
    std::size_t     minPortDistance           = 2UZ; // additional spacing for each port on a given side
    double          straightPenalty           = 1.2; // penalty for changing between vertical/horizontal
    double          diagonalPenalty           = 1.4; // penalty for changing between diagonal vs. non-diagonal
    std::size_t     outgoingDirectionLock     = 1UZ; // number of steps after src to maintain a direction before being allowed to change
    std::size_t     incomingDirectionLock     = 0UZ; // number of steps before dst to maintain a direction before being allowed to change
    std::size_t     optimiserMaxIterations    = 1024UZ;
    double          optimiserMinImprovement   = 0.01;
};

inline LayoutPref getLayoutPref(const std::shared_ptr<gr::BlockModel>& block) {
    auto& ui = block.get()->uiConstraints();
    if (auto it = ui.find("layout_pref"); it != ui.end()) {
        if (auto str = it->second.value_or(std::string_view{}); str.data() != nullptr) {
            if (str == "horizontal") {
                return LayoutPref::HORIZONTAL;
            }
            if (str == "vertical") {
                return LayoutPref::VERTICAL;
            }
            if (str == "auto") {
                return LayoutPref::AUTO;
            }
        }
    }
    return LayoutPref::UNDEFINED;
}

inline void setLayoutPref(std::shared_ptr<gr::BlockModel>& block, LayoutPref pref) {
    switch (pref) {
    case LayoutPref::HORIZONTAL: block->uiConstraints()["layout_pref"] = "horizontal"; break;
    case LayoutPref::VERTICAL: block->uiConstraints()["layout_pref"] = "vertical"; break;
    case LayoutPref::AUTO: block->uiConstraints()["layout_pref"] = "auto"; break;
    default: block->uiConstraints()["layout_pref"] = "undefined";
    }
}

template<gr::arithmetic_or_complex_like T = double>
Point<T> getPosition(const std::shared_ptr<gr::BlockModel>& block) {
    auto& ui = const_cast<gr::BlockModel*>(block.get())->uiConstraints();
    T     x  = Point<T>::invalid_position;
    T     y  = Point<T>::invalid_position;

    if (auto it = ui.find("pos_x"); it != ui.end()) {
        if (auto* val = it->second.get_if<double>()) {
            x = static_cast<T>(*val);
        }
    }
    if (auto it = ui.find("pos_y"); it != ui.end()) {
        if (auto* val = it->second.get_if<double>()) {
            y = static_cast<T>(*val);
        }
    }
    return {x, y};
}

// Port layout information storage
class PortInfo {
    bool                            _isInput;
    std::shared_ptr<gr::BlockModel> _block;         /// local reference to block -> used to access uiConstraints and other related infos
    std::size_t                     _idx;           /// ordinal of the port in the block
    std::optional<Side>             _preferredSide; /// UI related -> needs to be stored in BlockModel::uiConstraints() property_map
    Point<double>                   _position;      /// UI related -> needs to be stored in BlockModel::uiConstraints() property_map
    Direction                       _exitDir;       /// UI related -> needs to be stored in BlockModel::uiConstraints() property_map
    std::string                     _name;          /// existing information in BlockModel
    gr::port::BitMask               _portBitInfo;   /// existing information in BlockModel
    Style                           _style;         /// stored in BlockModel::uiConstraints()

public:
    static void mergeInto(gr::property_map& dst, const gr::property_map& src) {
        for (auto const& [k, v] : src) {
            dst[k] = v;
        }
    }

    void syncFromUIConstraints() {
        if (!_block) {
            return;
        }

        auto&      ui        = _block->uiConstraints();
        const auto mapKey    = _isInput ? "input_port_infos" : "output_port_infos";
        const auto portCount = _isInput ? _block->blockInputTypes().size() : _block->blockOutputTypes().size();

        // ensure vector exists and has correct size
        auto& vec = [&]() -> Tensor<pmt::Value>& {
            auto [it, _] = ui.try_emplace(mapKey, Tensor<pmt::Value>(extents_from, {portCount}));
            if (!it->second.holds<Tensor<pmt::Value>>()) {
                it->second = Tensor<pmt::Value>(gr::extents_from, {portCount});
            }
            auto& v = *it->second.get_if<Tensor<pmt::Value>>();
            if (v.size() != portCount) {
                v.resize({portCount}, gr::pmt::Value{});
            }
            return v;
        }();

        if (_idx >= vec.size()) {
            return;
        }

        // Ensure property_map exists at this index
        auto& m = [&]() -> gr::property_map& {
            if (!vec[_idx].holds<gr::property_map>()) {
                vec[_idx] = gr::property_map{};
            }
            return *vec[_idx].get_if<gr::property_map>();
        }();

        // --- Side (legacy int; keep key name "side") ---
        if (auto it = m.find("side"); it != m.end()) {
            if (auto* v = it->second.get_if<int>()) {
                _preferredSide = static_cast<Side>(*v);
            }
        } else {
            _preferredSide = _isInput ? Side::Left : Side::Right;
            m["side"]      = static_cast<int>(*_preferredSide);
        }

        // --- Position ---
        // Preferred: "pos" as array via Point<double> converter.
        // Legacy: "pos_x"/"pos_y" doubles.
        bool posSet = false;
        if (auto it = m.find("pos"); it != m.end()) {
            // Build a one-key map view to reuse fromPropertyMap(Point<double>)
            gr::property_map tmp;
            tmp["point"] = it->second;
            auto p       = gr::utf8::fromPropertyMap(Point<double>{}, tmp);
            if (p.x != 0.0 || p.y != 0.0) {
                _position = p;
                posSet    = true;
            }
        }
        if (!posSet) {
            bool hasX = false, hasY = false;
            if (auto it = m.find("pos_x"); it != m.end()) {
                if (auto* v = it->second.get_if<double>()) {
                    _position.x = *v;
                    hasX        = true;
                }
            }
            if (auto it = m.find("pos_y"); it != m.end()) {
                if (auto* v = it->second.get_if<double>()) {
                    _position.y = *v;
                    hasY        = true;
                }
            }
            if (!hasX || !hasY) {
                auto blockPos = getPosition<double>(_block);
                if (!hasX) {
                    _position.x = blockPos.x;
                }
                if (!hasY) {
                    _position.y = blockPos.y;
                }
            }
            // Write back new unified key "pos"
            mergeInto(m, gr::utf8::toPropertyMap(Point<double>{_position.x, _position.y}));
        }

        // --- Exit direction ---
        // Preferred: "direction" via enum string; Legacy: "exit_dir" int.
        if (auto it = m.find("direction"); it != m.end()) {
            gr::property_map tmp;
            tmp["direction"] = it->second;
            _exitDir         = gr::utf8::fromPropertyMap(Direction{}, tmp);
        } else if (auto it2 = m.find("exit_dir"); it2 != m.end()) {
            if (auto* v = it2->second.get_if<int>()) {
                _exitDir = static_cast<Direction>(*v);
            }
            // Write back new key
            mergeInto(m, gr::utf8::toPropertyMap(_exitDir));
        } else {
            // Default from side
            switch (*_preferredSide) {
            case Side::Left: _exitDir = Direction::West; break;
            case Side::Right: _exitDir = Direction::East; break;
            case Side::Top: _exitDir = Direction::North; break;
            case Side::Bottom: _exitDir = Direction::South; break;
            }
            mergeInto(m, gr::utf8::toPropertyMap(_exitDir));
        }

        // --- Style / Colour ---
        // Preferred: full Style keys (fg/bg/bold/…); Legacy: "colour" hex string.
        _style = gr::utf8::fromPropertyMap(Style{}, m);
        if (!_style.isSet()) {
            if (auto it = m.find("colour"); it != m.end()) {
                if (auto str = it->second.value_or(std::string_view{}); str.data() != nullptr) {
                    // interpret as fg hex
                    if (auto c = gr::utf8::color::parseHexRGB(str)) {
                        _style.fg    = Colour{c->r, c->g, c->b};
                        _style.fgSet = 1U;
                    }
                }
            }
        }
        // Emit unified Style keys (and keep legacy 'colour' if it existed)
        mergeInto(m, gr::utf8::toPropertyMap(_style));
        if (auto it = m.find("colour"); it == m.end() && _style.fgSet) {
            m["colour"] = gr::utf8::color::toHexRGB(_style.fg); // legacy convenience
        }
    }

    PortInfo() = delete;

    PortInfo(gr::PortDirection direction, const std::shared_ptr<gr::BlockModel>& srcRef, std::size_t index)
        : _isInput(direction == gr::PortDirection::INPUT), _block(srcRef), _idx(index), //
          _preferredSide(_isInput ? Side::Left : Side::Right), _exitDir(_isInput ? Direction::West : Direction::East) {
        const auto&                         types    = _isInput ? _block->blockInputTypes() : _block->blockOutputTypes();
        const std::vector<gr::PortMetaInfo> metaInfo = _isInput ? _block->inputMetaInfos() : _block->outputMetaInfos();

        _name        = metaInfo[_idx].name;
        _portBitInfo = types[_idx];
        syncFromUIConstraints();
    }
    PortInfo(const PortInfo&)            = default;
    PortInfo(PortInfo&&)                 = default;
    PortInfo& operator=(const PortInfo&) = default;
    PortInfo& operator=(PortInfo&&)      = default;

    [[nodiscard]] std::shared_ptr<gr::BlockModel>      block() noexcept { return _block; }
    [[nodiscard]] bool                                 drawLabel() const noexcept { return _block->uiConstraints().find("draw_label") != _block->uiConstraints().end(); }
    [[nodiscard]] std::size_t                          portIdx() const noexcept { return _idx; }
    [[nodiscard]] constexpr gr::port::BitMask          portBitInfo() const noexcept { return _portBitInfo; }
    [[nodiscard]] constexpr std::string_view           name() const noexcept { return drawLabel() ? (_name.empty() ? utf8_digits(portIdx(), isSynchronous()) : std::string_view(_name)) : ""; }
    [[nodiscard]] constexpr gr::PortDirection          direction() const { return gr::port::decodeDirection(_portBitInfo); }
    [[nodiscard]] constexpr bool                       isInput() const noexcept { return gr::port::isInput(_portBitInfo); }
    [[nodiscard]] constexpr bool                       isSynchronous() const noexcept { return gr::port::isSynchronous(_portBitInfo); }
    [[nodiscard]] constexpr bool                       isStream() const noexcept { return gr::port::isStream(_portBitInfo); }
    [[nodiscard]] constexpr const std::optional<Side>& preferredSide() const noexcept { return _preferredSide; }
    template<gr::arithmetic_or_complex_like T>
    [[nodiscard]] constexpr Point<T> anchorPoint() const noexcept {
        return position().get<T>();
    }
    template<gr::arithmetic_or_complex_like T>
    [[nodiscard]] constexpr Point<T> exitPoint() const noexcept {
        Point<T> anchor = anchorPoint<T>();
        using enum Direction;
        switch (exitDir()) {
        case West: return {anchor.x > T(0) ? anchor.x - T(1) : T(0), anchor.y};
        case East: return {anchor.x + T(1), anchor.y};
        case North: return {anchor.x, anchor.y > 0 ? anchor.y - T(1) : T(0)};
        case South: return {anchor.x, anchor.y + T(1)};
        default: return anchor;
        }
    }

    [[nodiscard]] constexpr const Point<double>& position() const noexcept { return _position; }
    void                                         addPosition(Point<double> offset) noexcept { position(position() + offset); }
    [[nodiscard]] constexpr const Direction&     exitDir() const noexcept { return _exitDir; }
    [[nodiscard]] constexpr Style                style() const noexcept { return _style; }

    void preferredSide(Side newSide) noexcept {
        _preferredSide = newSide;
        updateUIConstraints("side", gr::pmt::Value(static_cast<int>(newSide)));
    }

    void position(Point<double> newPos) noexcept {
        _position = newPos;
        // write unified "pos"
        updateUIConstraints(gr::utf8::toPropertyMap(Point<double>{_position.x, _position.y}));
        // keep legacy too (optional; remove if you want to stop emitting them)
        updateUIConstraints("pos_x", gr::pmt::Value(_position.x));
        updateUIConstraints("pos_y", gr::pmt::Value(_position.y));
    }

    void exitDir(Direction newDir) noexcept {
        _exitDir = newDir;
        updateUIConstraints(gr::utf8::toPropertyMap(_exitDir)); // writes "direction"
    }

    void style(Style style) noexcept {
        _style = style;
        updateUIConstraints(gr::utf8::toPropertyMap(_style)); // fg/bg/bold/…
        // also maintain simple legacy hex for tooling that still reads it
        if (_style.fgSet) {
            updateUIConstraints("colour", gr::pmt::Value(gr::utf8::color::toHexRGB(_style.fg)));
        }
    }

private:
    void updateUIConstraints(const gr::property_map& patch) {
        if (!_block) {
            return;
        }

        auto&      ui        = _block->uiConstraints();
        const auto mapKey    = _isInput ? "input_port_infos" : "output_port_infos";
        const auto portCount = _isInput ? _block->blockInputTypes().size() : _block->blockOutputTypes().size();

        auto& vec = [&]() -> Tensor<pmt::Value>& {
            auto [it, _] = ui.try_emplace(mapKey, Tensor<pmt::Value>(extents_from, {portCount}));
            if (!it->second.holds<Tensor<pmt::Value>>()) {
                it->second = Tensor<pmt::Value>(extents_from, {portCount});
            }
            auto& v = *it->second.get_if<Tensor<pmt::Value>>();
            if (v.size() != portCount) {
                v.resize({portCount}, pmt::Value());
            }
            return v;
        }();

        if (_idx >= vec.size()) {
            return;
        }

        if (!vec[_idx].holds<gr::property_map>()) {
            vec[_idx] = gr::property_map{};
        }

        auto& m = *vec[_idx].get_if<gr::property_map>();
        mergeInto(m, patch);
    }

    void updateUIConstraints(const std::string& key, pmt::Value value) {
        if (!_block) {
            return;
        }

        auto&      ui        = _block->uiConstraints();
        const auto mapKey    = _isInput ? "input_port_infos" : "output_port_infos";
        const auto portCount = _isInput ? _block->blockInputTypes().size() : _block->blockOutputTypes().size();

        // Ensure vector exists and has correct size
        auto& vec = [&]() -> Tensor<pmt::Value>& {
            auto [it, _] = ui.try_emplace(mapKey, Tensor<pmt::Value>(extents_from, {portCount}));
            if (!it->second.holds<Tensor<pmt::Value>>()) {
                it->second = Tensor<pmt::Value>(extents_from, {portCount});
            }
            auto& v = *it->second.get_if<Tensor<pmt::Value>>();
            if (v.size() != portCount) {
                v.resize({portCount}, pmt::Value());
            }
            return v;
        }();

        // Ensure property_map exists at this index
        if (_idx >= vec.size()) {
            return; // Safety check
        }

        if (!vec[_idx].holds<gr::property_map>()) {
            vec[_idx] = gr::property_map{};
        }

        auto& m                       = *vec[_idx].get_if<gr::property_map>();
        m[convert_string_domain(key)] = std::move(value);
    }
};

template<gr::PortDirection direction>
inline std::vector<PortInfo> getPortInfos(const std::shared_ptr<gr::BlockModel>& block) {
    constexpr bool        isInput = direction == gr::PortDirection::INPUT;
    const auto&           types   = isInput ? block->blockInputTypes() : block->blockOutputTypes();
    std::vector<PortInfo> out;
    out.reserve(types.size());

    for (std::size_t i = 0UZ; i < types.size(); ++i) {
        out.emplace_back(PortInfo(direction, block, i));
    }
    return out;
}

inline std::size_t inputCount(const std::shared_ptr<gr::BlockModel>& block) { return getPortInfos<gr::PortDirection::INPUT>(block).size(); }
inline std::size_t outputCount(const std::shared_ptr<gr::BlockModel>& block) { return getPortInfos<gr::PortDirection::OUTPUT>(block).size(); }

inline std::size_t portsOnSide(const std::shared_ptr<gr::BlockModel>& block, Side side) {
    std::size_t count = 0UZ;
    for (const auto& info : getPortInfos<gr::PortDirection::INPUT>(block)) {
        if (info.preferredSide() && *info.preferredSide() == side) {
            count++;
        }
    }
    for (const auto& info : getPortInfos<gr::PortDirection::OUTPUT>(block)) {
        if (info.preferredSide() && *info.preferredSide() == side) {
            count++;
        }
    }
    return count;
}

template<gr::PortDirection direction>
PortInfo getPortByDefinition(const std::shared_ptr<gr::BlockModel>& block, const gr::PortDefinition& portDef, std::source_location loc = std::source_location::current()) {
    std::vector<PortInfo> infos   = getPortInfos<direction>(block);
    std::size_t           portIdx = gr::absolutePortIndex<direction>(block, portDef, loc);
    return infos[portIdx];
}

template<gr::arithmetic_or_complex_like T = std::size_t>
T calculateWidth(const std::shared_ptr<gr::BlockModel>& block) {
    using enum Side;

    auto sumLabel = [&](Side side) -> T {
        T acc = T(0);
        for (const auto& info : getPortInfos<gr::PortDirection::INPUT>(block)) {
            if (info.preferredSide() && *info.preferredSide() == side) {
                acc = std::max(acc, static_cast<T>(gr::utf8::length(info.name())));
            }
        }
        for (const auto& info : getPortInfos<gr::PortDirection::OUTPUT>(block)) {
            if (info.preferredSide() && *info.preferredSide() == side) {
                acc = std::max(acc, static_cast<T>(gr::utf8::length(info.name())));
            }
        }
        return acc;
    };

    const T leftW  = sumLabel(Left);
    const T rightW = sumLabel(Right);

    const T lrBase   = leftW + rightW + T(1);
    const T titWidth = static_cast<T>(gr::utf8::length(block->name()) + 2UZ);
    T       interior = std::max(lrBase, titWidth);

    const bool hasTB = portsOnSide(block, Top) + portsOnSide(block, Bottom) > 0UZ;
    if (hasTB) {
        auto rowLen = [&](Side side) {
            T acc = T(0);
            for (const auto& info : getPortInfos<gr::PortDirection::INPUT>(block)) {
                if (info.preferredSide() && *info.preferredSide() == side) {
                    acc += T(1) + static_cast<T>(gr::utf8::length(info.name())) + T(1);
                }
            }
            for (const auto& info : getPortInfos<gr::PortDirection::OUTPUT>(block)) {
                if (info.preferredSide() && *info.preferredSide() == side) {
                    acc += T(1) + static_cast<T>(gr::utf8::length(info.name())) + T(1);
                }
            }
            return acc;
        };
        const T topLen = rowLen(Top);
        const T botLen = rowLen(Bottom);

        const T rowMax      = std::max(topLen > T(0) ? topLen - T(1) : T(0), botLen > T(0) ? botLen - T(1) : T(0));
        const T lrPlusTitle = leftW + rightW + static_cast<T>(2UZ + gr::utf8::length(block->name()));

        interior = std::max({interior, rowMax, lrPlusTitle});
    }

    return interior + 2;
}

template<gr::arithmetic_or_complex_like T = std::size_t>
T calculateHeight(const std::shared_ptr<gr::BlockModel>& block) {
    return std::max(static_cast<T>(std::max(portsOnSide(block, Side::Left), portsOnSide(block, Side::Right))) + T(2), T(3));
}

template<gr::arithmetic_or_complex_like T>
Point<T> centre(const std::shared_ptr<gr::BlockModel>& block) {
    Point<T> pos    = getPosition<T>(block);
    T        width  = calculateWidth<T>(block);
    T        height = calculateHeight<T>(block);
    return {pos.x + width / T(2), pos.y + height / T(2)};
}

template<gr::arithmetic_or_complex_like T>
Point<T> topLeft(const std::shared_ptr<gr::BlockModel>& block) {
    return getPosition<T>(block);
}

template<gr::arithmetic_or_complex_like T>
Point<T> topRight(const std::shared_ptr<gr::BlockModel>& block) {
    Point<T> pos   = getPosition<T>(block);
    T        width = calculateWidth<T>(block);
    return {pos.x + width - T(1), pos.y};
}

template<gr::arithmetic_or_complex_like T>
Point<T> bottomLeft(const std::shared_ptr<gr::BlockModel>& block) {
    Point<T> pos    = getPosition<T>(block);
    T        height = calculateHeight<T>(block);
    return {pos.x, pos.y + height - T(1)};
}

template<gr::arithmetic_or_complex_like T>
Point<T> bottomRight(const std::shared_ptr<gr::BlockModel>& block) {
    Point<T> pos    = getPosition<T>(block);
    T        width  = calculateWidth<T>(block);
    T        height = calculateHeight<T>(block);
    return {pos.x + width - T(1), pos.y + height - T(1)};
}

template<gr::arithmetic_or_complex_like T>
Point<T> size(const std::shared_ptr<gr::BlockModel>& block) {
    return {calculateWidth<T>(block), calculateHeight<T>(block)};
}

// Update block position and layout ports
inline void updateBlock(std::shared_ptr<gr::BlockModel>& block, double x, double y) {
    block->uiConstraints()["pos_x"] = x;
    block->uiConstraints()["pos_y"] = y;
    // Store calculated dimensions
    block->uiConstraints()["width"]  = calculateWidth<double>(block);
    block->uiConstraints()["height"] = calculateHeight<double>(block);

    // init port ui meta info to defaults
    auto inputInfos   = getPortInfos<gr::PortDirection::INPUT>(block);
    auto outputInfos  = getPortInfos<gr::PortDirection::OUTPUT>(block);
    auto infos        = std::array{std::views::all(inputInfos), std::views::all(outputInfos)};
    auto allPortInfos = infos | std::views::join;

    // Layout ports
    LayoutPref layout = getLayoutPref(block);
    using enum Side;
    std::array<std::size_t, 4UZ> offsets{1UZ, 1UZ, 1UZ, 1UZ};
    for (auto& portInfo : allPortInfos) {
        if (!portInfo.preferredSide().has_value()) {
            switch (layout) {
            case LayoutPref::VERTICAL: portInfo.preferredSide(portInfo.direction() == gr::PortDirection::INPUT ? Top : Bottom); break;
            case LayoutPref::HORIZONTAL:
            case LayoutPref::AUTO:
            case LayoutPref::UNDEFINED:
            default: portInfo.preferredSide(portInfo.direction() == gr::PortDirection::INPUT ? Left : Right); break;
            }
        }

        std::size_t sideIdx = std::to_underlying(*portInfo.preferredSide());
        switch (portInfo.preferredSide().value()) {
        case Left:
            portInfo.position(topLeft<double>(block) + Point<double>(0.0, static_cast<double>(offsets[sideIdx])));
            portInfo.exitDir(Direction::West);
            break;
        case Right:
            portInfo.position(topRight<double>(block) + Point<double>(0.0, static_cast<double>(offsets[sideIdx])));
            portInfo.exitDir(Direction::East);
            break;
        case Top:
            portInfo.position(topLeft<double>(block) + Point<double>(static_cast<double>(offsets[sideIdx]), 0.0));
            offsets[sideIdx] += gr::utf8::length(portInfo.name());
            portInfo.exitDir(Direction::North);
            break;
        case Bottom:
            portInfo.position(bottomLeft<double>(block) + Point<double>(static_cast<double>(offsets[sideIdx]), 0.0));
            offsets[sideIdx] += gr::utf8::length(portInfo.name());
            portInfo.exitDir(Direction::South);
            break;
        }
        offsets[sideIdx]++;
    }
}

template<gr::arithmetic_or_complex_like T>
void updateBlock(std::shared_ptr<gr::BlockModel>& block, Point<T> newPosition) {
    updateBlock(block, static_cast<double>(newPosition.x), static_cast<double>(newPosition.y));
}

// edge helpers
[[nodiscard]] inline EdgeType getEdgeType(const gr::Edge& edge) {
    auto& ui = const_cast<gr::Edge&>(edge).uiConstraints();
    if (auto it = ui.find("edge_type"); it != ui.end()) {
        if (auto str = it->second.value_or(std::string_view{}); str.data() != nullptr) {
            if (str == "feedback") {
                return EdgeType::Feedback;
            }
            if (str == "lateral") {
                return EdgeType::Lateral;
            }
        }
    }
    return EdgeType::Forward;
}

inline void setEdgeType(gr::Edge& edge, EdgeType type) {
    std::string typeStr               = type == EdgeType::Feedback ? "feedback" : type == EdgeType::Lateral ? "lateral" : "forward";
    edge.uiConstraints()["edge_type"] = typeStr;
}

[[nodiscard]] inline Style style(const std::shared_ptr<gr::BlockModel>& b) noexcept { return gr::utf8::fromPropertyMap(Style{}, b->uiConstraints()); }

[[nodiscard]] inline Style style(const gr::Edge& edge) noexcept { return gr::utf8::fromPropertyMap(Style{}, edge.uiConstraints()); }

inline void style(const std::shared_ptr<gr::BlockModel>& block, Style style) {
    auto&      ui = block->uiConstraints();
    const auto m  = gr::utf8::toPropertyMap(style); // → { "colour" : "#RRGGBB" }
    for (const auto& [k, v] : m) {
        ui[k] = v;
    }
}
inline void colour(const std::shared_ptr<gr::BlockModel>& block, Colour col) { style(block, Style{.fg = col, .fgSet = true}); }

inline void style(gr::Edge edge, Style style) {
    auto&      ui = edge.uiConstraints();
    const auto m  = gr::utf8::toPropertyMap(style);
    for (const auto& [k, v] : m) {
        ui[k] = v;
    }
}
inline void colour(gr::Edge edge, Colour col) { style(edge, Style{.fg = {col}, .fgSet = true}); }

template<gr::PortDirection direction>
void style(const std::shared_ptr<gr::BlockModel>& block, gr::PortDefinition portDefinition, Style style, std::source_location loc = std::source_location::current()) {
    std::vector<PortInfo> infos   = getPortInfos<direction>(block);
    std::size_t           portIdx = gr::absolutePortIndex<direction>(block, portDefinition, loc);
    infos.at(portIdx).style(style);
}

template<gr::PortDirection direction>
void colour(const std::shared_ptr<gr::BlockModel>& block, gr::PortDefinition portDefinition, Colour col, std::source_location loc = std::source_location::current()) {
    style<direction>(block, portDefinition, Style{.fg = col, .fgSet = true}, loc);
}

template<gr::arithmetic_or_complex_like T>
Point<T> minRequiredGapDistance(const std::shared_ptr<gr::BlockModel>& nodeFrom, const std::shared_ptr<gr::BlockModel>& nodeTo, const LayoutPreference& cfg = {}) noexcept {
    using enum Side;
    const Point centreA = centre<T>(nodeFrom);
    const Point centreB = centre<T>(nodeTo);

    auto minDistX = static_cast<T>(cfg.minGap + cfg.minPortDistance * (centreA.x < centreB.x ? (portsOnSide(nodeFrom, Right) + portsOnSide(nodeTo, Left)) : (portsOnSide(nodeFrom, Left) + portsOnSide(nodeTo, Right))));

    auto minDistY = static_cast<T>(cfg.minGap + cfg.minPortDistance * (centreA.y < centreB.y ? (portsOnSide(nodeFrom, Bottom) + portsOnSide(nodeTo, Top)) : (portsOnSide(nodeFrom, Top) + portsOnSide(nodeTo, Bottom))));

    return {minDistX, minDistY};
}

template<gr::arithmetic_or_complex_like T>
void flipAllPortSides(std::shared_ptr<gr::BlockModel>& block, std::source_location loc = std::source_location::current()) {
    auto infos = std::array{std::views::all(getPortInfos<gr::PortDirection::INPUT>(block)), std::views::all(getPortInfos<gr::PortDirection::OUTPUT>(block))};
    for (auto& info : infos | std::views::join) {
        if (!info.preferredSide()) {
            throw gr::exception(std::format("node: {}-{} undefined port side", block->name(), info.name()), loc);
        }
        info.preferredSide(oppositeSide(*info.preferredSide()));
    }

    updateBlock(block, getPosition<T>(block));
}

template<gr::arithmetic_or_complex_like T>
T edgeCost(const gr::Edge& edge, std::source_location loc = std::source_location::current()) {
    PortInfo srcPort = getPortByDefinition<gr::PortDirection::OUTPUT>(edge.sourceBlock(), edge.sourcePortDefinition(), loc);
    PortInfo dstPort = getPortByDefinition<gr::PortDirection::INPUT>(edge.destinationBlock(), edge.destinationPortDefinition(), loc);

    return manhattanNorm(srcPort.exitPoint<T>(), dstPort.exitPoint<T>());
}

template<gr::arithmetic_or_complex_like T>
T totalRoutingCost(const std::span<std::shared_ptr<gr::BlockModel>>& nodes, const std::span<const gr::Edge>& edges, std::source_location loc = std::source_location::current()) {
    std::ranges::for_each(nodes, [](std::shared_ptr<gr::BlockModel>& n) { updateBlock(n, getPosition<T>(n)); });
    return std::accumulate(std::ranges::begin(edges), std::ranges::end(edges), T(0), [&](T acc, const gr::Edge& e) { return acc + edgeCost<T>(e, loc); });
}

// sum of costs of edges incident to node nodeIndex
template<gr::arithmetic_or_complex_like T>
T incidentCost(const std::span<const gr::Edge>& edges, std::shared_ptr<gr::BlockModel> nodeIndex, std::source_location loc = std::source_location::current()) {
    updateBlock(nodeIndex, getPosition<T>(nodeIndex));
    T acc = T(0);
    for (const gr::Edge& e : edges) {
        if (e.sourceBlock() == nodeIndex || e.destinationBlock() == nodeIndex) {
            acc += edgeCost<T>(e, loc);
        }
    }
    return acc;
}

template<gr::arithmetic_or_complex_like T>
[[maybe_unused]] std::size_t optimiseSideFlips(gr::Graph& graph, std::size_t maxIters = 0, double improveEps = 0.10, std::source_location loc = std::source_location::current()) {
    using enum Side;

    if (maxIters == 0) {
        maxIters = std::max<std::size_t>(1, graph.blocks().size());
    }
    std::ranges::for_each(graph.blocks(), [](std::shared_ptr<gr::BlockModel>& n) { updateBlock(n, getPosition<T>(n)); });

    std::size_t flippedPorts = false;
    T           prevTotal    = totalRoutingCost<T>(graph.blocks(), graph.edges(), loc);
    for (std::size_t it = 0; it < maxIters; ++it) {
        bool anyImprove = false;

        for (std::shared_ptr<gr::BlockModel>& n : graph.blocks()) {
            auto in  = getPortInfos<gr::PortDirection::INPUT>(n);
            auto out = getPortInfos<gr::PortDirection::OUTPUT>(n);
            if (getLayoutPref(n) == LayoutPref::HORIZONTAL || getLayoutPref(n) == LayoutPref::VERTICAL) {
                const T before = incidentCost<T>(graph.edges(), n);
                flipAllPortSides<T>(n, loc);
                if (const T after = incidentCost<T>(graph.edges(), n); after < before) { // better
                    anyImprove = true;
                    flippedPorts++;
                } else { // worse -> revert port flip
                    flipAllPortSides<T>(n, loc);
                }
                continue;
            }

            //  Snapshot current assignment
            const std::size_t                I = n->inputMetaInfos().size();
            const std::size_t                O = n->outputMetaInfos().size();
            std::vector<std::optional<Side>> keepIn(I);
            std::vector<std::optional<Side>> keepOut(O);
            for (std::size_t i = 0; i < I; ++i) {
                keepIn[i] = in[i].preferredSide();
            }
            for (std::size_t i = 0; i < O; ++i) {
                keepOut[i] = out[i].preferredSide();
            }

            // ---------------------- AUTO: enumerate legal assignments ----------------------
            const T                              keep = incidentCost<T>(graph.edges(), n);
            T                                    best = keep;
            static constexpr std::array<Side, 4> all{Left, Right, Top, Bottom};

            // Working buffers
            std::vector<Side> curIn(I);
            std::vector<Side> curOut(O);
            // Best candidate to commit later
            std::vector<Side> bestIn(I);
            std::vector<Side> bestOut(O);
            // Initialise the best candidate with the current assignment (defined by keepIn/keepOut)
            for (std::size_t i = 0; i < I; ++i) {
                bestIn[i] = keepIn[i].value_or(Left);
            }
            for (std::size_t i = 0; i < O; ++i) {
                bestOut[i] = keepOut[i].value_or(Right);
            }

            // Limit brute force: 4^(I+O) := 2^(2*(I+O))
            const std::size_t stateCount = static_cast<std::size_t>(1) << (2u * static_cast<unsigned>(I + O));
            if (stateCount > 200'000u) {
                // Too many states: skip exhaustive search (keep current)
                continue;
            }

            // DFS over inputs (we carry a side-bitmask for inputs); outputs can’t reuse these sides
            std::function<void(std::size_t, std::uint8_t)> dfsIn;
            std::function<void(std::size_t, std::uint8_t)> dfsOut;
            dfsOut = [&](std::size_t j, std::uint8_t inMask) {
                if (j == O) {
                    // apply candidate
                    for (std::size_t i = 0; i < I; ++i) {
                        in[i].preferredSide(curIn[i]);
                    }
                    for (std::size_t i = 0; i < O; ++i) {
                        out[i].preferredSide(curOut[i]);
                    }
                    updateBlock(n, getPosition<T>(n));
                    const T c = incidentCost<T>(graph.edges(), n);
                    if (c < best) {
                        best       = c;
                        bestIn     = curIn;
                        bestOut    = curOut;
                        anyImprove = true;
                        flippedPorts++;
                    }
                    // restore is not needed here; the next iteration will overwrite curIn/curOut
                    return;
                }
                for (Side s : all) {
                    const std::uint8_t bit = static_cast<std::uint8_t>(1u << static_cast<std::uint8_t>(std::to_underlying(s)));
                    if (inMask & bit) {
                        continue; // would mix IO on the same side
                    }
                    curOut[j] = s;
                    dfsOut(j + 1, inMask);
                }
            };

            dfsIn = [&](std::size_t i, std::uint8_t mask) {
                if (i == I) {
                    dfsOut(0, mask);
                    return;
                }
                for (Side s : all) {
                    curIn[i]           = s;
                    const unsigned bit = 1u << static_cast<unsigned>(std::to_underlying(s));
                    dfsIn(i + 1, static_cast<std::uint8_t>(mask | bit));
                }
            };

            // Run enumeration
            dfsIn(0, 0);

            // Commit best candidate if improved
            if (best < keep) {
                for (std::size_t i = 0; i < I; ++i) {
                    in[i].preferredSide(bestIn[i]);
                }
                for (std::size_t i = 0; i < O; ++i) {
                    out[i].preferredSide(bestOut[i]);
                }
                updateBlock(n, getPosition<T>(n));
            } else {
                // explicitly restore (defensive)
                for (std::size_t i = 0; i < I; ++i) {
                    in[i].preferredSide(*keepIn[i]);
                }
                for (std::size_t i = 0; i < O; ++i) {
                    out[i].preferredSide(*keepOut[i]);
                }
                updateBlock(n, getPosition<T>(n));
            }
        }

        const T      tot = totalRoutingCost<T>(graph.blocks(), graph.edges(), loc);
        const double rel = (prevTotal == T(0)) ? 0.0 : static_cast<double>(prevTotal - tot) / static_cast<double>(prevTotal);
        if (!anyImprove || rel < improveEps) {
            break;
        }
        prevTotal = tot;
    }

    return flippedPorts;
}

// -----------------------------------------------------------------------------
// Downstream "string-of-pearls" weight with fractional increment 1/N_in(child).
// Sinks/isolated start at 1.0. For u→v we consider cand = dfs(v) + 1/N_in(v),
// and take the maximum over all outgoing edges (side branches don’t inflate).
// Cycles are skipped via an 'onStack' mask.
// -----------------------------------------------------------------------------
inline std::vector<double> computeEffectiveWeights(const gr::Graph& graph, const std::span<const gr::Edge>& edges) {
    std::vector<std::vector<std::size_t>> out(graph.blocks().size());
    std::vector<std::size_t>              connectedChildren(graph.blocks().size(), 0UZ);

    for (auto const& edge : edges) {
        out[*gr::graph::blockIndex(graph, edge.sourceBlock())].push_back(*gr::graph::blockIndex(graph, edge.destinationBlock()));
        connectedChildren[*gr::graph::blockIndex(graph, edge.destinationBlock())]++; // N.B. node can have multiple children connected to the same output.
    }

    std::vector<double> effectiveMass(graph.blocks().size(), std::numeric_limits<double>::quiet_NaN()); // NaN = not computed yet
    std::vector<bool>   onStack(graph.blocks().size(), false);

    std::function<double(std::size_t)> dfs = [&](std::size_t u) -> double {
        if (!std::isnan(effectiveMass[u])) {
            return effectiveMass[u];
        }

        onStack[u]  = true;
        double best = 1.0; // ensure at least minimal mass unit
        for (std::size_t nodeIdx : out[u]) {
            assert(nodeIdx < graph.blocks().size());
            if (onStack[nodeIdx]) { // detected cycle -> skip
                continue;
            }
            const double inc = connectedChildren[nodeIdx] > 0 ? 1.0 / static_cast<double>(connectedChildren[nodeIdx]) : 0.0;
            best             = std::max(best, dfs(nodeIdx) + inc);
        }
        onStack[u]              = false;
        return effectiveMass[u] = best;
    };

    std::vector<double> w(graph.blocks().size(), 1.0);
    for (std::size_t i = 0UZ; i < graph.blocks().size(); ++i) {
        w[i] = dfs(i);
    }
    return w;
}

/** ******************************************************************/
/** ******************************************************************/
/** ******************************************************************/
/** ******************************************************************/

/**
 * @brief phase1_place_spring_model Unified notes for classic force-directed graph layout methods.
 *
 * This block collects the common notation used by spring/force layouts and
 * then specializes it to three seminal methods: Eades (1984),
 * Fruchterman–Reingold (1991), and Tutte’s barycentric embedding (1963).
 *
 * Let G = (V, E) with |V| = n. Each vertex u ∈ V has a position x_u ∈ ℝ².
 * For a pair (u, v) define:
 *   d(u,v) = ‖x_u − x_v‖₂        // Euclidean distance
 *   ẑ(u,v) = (x_u − x_v) / d(u,v) // unit direction from v to u (guard d>0)
 *
 * Two generic force components are used:
 *   • Repulsion  f_rep(u,v) acting on (unordered) pairs u ≠ v
 *   • Attraction f_attr(u,v) acting on edges (u,v) ∈ E
 *
 * Forces are radial by default: F_rep(u,v) = f_rep(d) · ẑ(u,v),
 * F_attr(u,v) = − f_attr(d) · ẑ(u,v)  (pulling neighbors together).
 *
 * Typical iterative scheme (temperature-limited step; Euler or Verlet):
 *
 *   // At iteration t with temperature T(t):
 *   for (u in V) F[u] = (0,0)
 *   // repulsion (all-pairs or an approximation like Barnes–Hut):
 *   for (unordered u≠v) accumulate F[u] +=  f_rep(d(u,v)) · ẑ(u,v)
 *                                  F[v] +=  f_rep(d(v,u)) · ẑ(v,u)
 *   // attraction (edges only):
 *   for ((u,v) in E)   accumulate F[u] += -f_attr(d(u,v)) · ẑ(u,v)
 *                                  F[v] +=  f_attr(d(u,v)) · ẑ(u,v)
 *   // optional: gravity/centering/tethers; clamp large |F[u]| if needed
 *   // integrate & cool:
 *   for (u in V) x_u ← x_u + step(F[u], T(t));   T(t+1) ← cool(T(t))
 *
 * Popular choices for step(·) include:
 *   • Euler with a cap: Δx_u = clamp(F[u], T(t))
 *   • Damped Euler:     v_u = α v_u + β F[u],  x_u += v_u
 *   • (Fast) Verlet:    x_u(t+1) = x_u(t) + (1-γ)(x_u(t) - x_u(t-1)) + a_u·Δt²
 *
 * ────────────────────────────────────────────────────────────────────────────
 * Eades (1984): log-spring attraction + inverse-square repulsion
 * ────────────────────────────────────────────────────────────────────────────
 * Idea: model edges as (soft) springs whose force grows like log(d / L),
 * while non-neighbors repel with an inverse-square law.
 *
 * Parameters:
 *   k_a > 0 (attractive gain), k_r > 0 (repulsive gain), L > 0 (length scale).
 *
 * Forces (radial magnitudes in terms of d = d(u,v)):
 *   f_attr(d) = k_a · ln(d / L)                // edges only
 *   f_rep(d)  = k_r / d²                        // all unordered pairs
 *
 * Practical notes:
 *   • Clamp the argument of ln(·) with a small ε to avoid d≈0 singularities.
 *   • Use a cooling schedule or a per-iteration displacement cap to stabilize.
 *   • L sets the “typical” edge length; tune k_a:k_r for compactness vs. spread.
 *   • simple, expressive parameters (k_a, k_r, L);
 *   • good for small/medium graphs;
 *   • can need careful temperature control to avoid oscillations.
 *
 * Reference:
 *   P. Eades, “A heuristic for graph drawing,” Congressus Numerantium, vol. 42, pp. 149–160, 1984.
 *
 * ────────────────────────────────────────────────────────────────────────────
 * Fruchterman–Reingold (1991): scale-free k with 1/d vs. d²/k forces
 * ────────────────────────────────────────────────────────────────────────────
 * Idea: choose a global “ideal” distance k that depends on area A and n,
 * then use simple inverse/ quadratic laws whose units cancel via k.
 *
 * Canonical choice of k:
 *   k = C · sqrt(A / n)   with C ≈ 1 (tunable), A = drawing area
 *
 * Forces (radial magnitudes; d = d(u,v)):
 *   f_rep(d)  =  k² / d                 // all unordered pairs
 *   f_attr(d) =  d² / k                  // edges only
 *
 * Practical notes:
 *   • Cooling matters: T(t) typically decreases linearly or exponentially.
 *   • Often augmented with mild gravity/centering to reduce drift.
 *   • Barnes–Hut or grid hashing is commonly used to accelerate f_rep.
 *   • scale-free via k; widely used baseline;
 *   • pairwise repulsion benefits from BH or grid acceleration for large n.
 *
 * Reference (publisher PDF link):
 *   T. M. J. Fruchterman and E. M. Reingold, “Graph drawing by force-directed
 *   placement,” Software: Practice and Experience, vol. 21, no. 11,
 *   pp. 1129–1164, 1991. PDF: https://onlinelibrary.wiley.com/doi/pdf/10.1002/spe.4380211102
 *
 * ────────────────────────────────────────────────────────────────────────────
 * Tutte (1963): barycentric (spring) embedding with fixed outer face
 * ────────────────────────────────────────────────────────────────────────────
 * Idea: NOT an iterative force method over all pairs. Fix the vertices of a
 * chosen outer face on the boundary of a convex polygon; place every interior
 * vertex at the average (barycenter) of its neighbors. This solves a sparse,
 * strictly convex quadratic energy → unique straight-line embedding that is
 * planar for 3-connected planar graphs.
 *
 * Barycentric equations (for interior v with neighbors N(v)):
 *   x_v = (1/deg(v)) · Σ_{u∈N(v)} x_u
 *   y_v = (1/deg(v)) · Σ_{u∈N(v)} y_u
 * Equivalently, solve L_II X_I = −L_IB X_B for interior coordinates X_I,
 * where L is the graph Laplacian and B the fixed boundary vertices.
 *
 * Force view (for intuition only): edges behave like identical linear springs
 * with zero rest length, but boundary pins replace explicit repulsion.
 * Exact convex embedding for planar, 3-connected graphs with a chosen
 * outer face; no pairwise repulsion needed; solves a linear system once.
 *
 * Reference (publisher PDF link):
 *   W. T. Tutte, “How to draw a graph,” Proc. London Math. Soc.,
 *   s3-13(1), pp. 743–767, 1963. PDF: https://londmathsoc.onlinelibrary.wiley.com/doi/pdf/10.1112/plms/s3-13.1.743
 */
inline static void layoutSpringModel(gr::Graph& graph, const LayoutPreference& cfg = {}, std::source_location loc = std::source_location::current()) {
    using T = double;
    using enum Side;

    const size_t N = graph.blocks().size();
    if (N == 0) {
        return;
    }

    // 0) normalise layout + give every node a valid position and anchors
    for (auto& n : graph.blocks()) {
        if (getLayoutPref(n) == LayoutPref::UNDEFINED) {
            setLayoutPref(n, cfg.layout == LayoutPref::AUTO ? LayoutPref::HORIZONTAL : cfg.layout);
        }
        if (getPosition<T>(n) == Point<T>::undefined()) {
            updateBlock(n, static_cast<T>(cfg.minMargin), static_cast<T>(cfg.minMargin));
        } else {
            updateBlock(n, getPosition<T>(n));
        }
    }

    // 2) right-weights (directed, downstream-only, fractional 1/N_in)
    const std::vector<double> effectiveWeights = computeEffectiveWeights(graph, graph.edges());

    // 3) initialise continuous state
    std::vector<Point<T>> position(N);
    std::vector<Point<T>> prevPosition(N);
    std::vector<Point<T>> velocity(N, {static_cast<T>(0), static_cast<T>(0)});
    for (std::size_t i = 0UZ; i < N; ++i) {
        position[i]     = getPosition<T>(graph.blocks()[i]);
        prevPosition[i] = position[i];
        // position[i] = {T(i), T(i)};
    }

    // 4) constants (tuned to be stable + compact)

    [[maybe_unused]] constexpr double kSrcTether = 0.1;
    [[maybe_unused]] constexpr double kSnkTether = 0.1;
    constexpr double                  kDrag      = 0.2;  // usually [0.02, 0.06]
    constexpr double                  gravUp     = 0.01; // constant upward pull (all nodes; negative y)
    constexpr double                  gravRight  = 0.1;  // right pull scaled by wRight[i]
    constexpr double                  gravLeftW  = 0.06; // weak left bias for all (keeps chain taut to sources)

    constexpr double      kStep         = 0.05;     // semi-implicit Euler step
    constexpr double      kDampening    = 0.85;     // Euler integrator velocity damping
    constexpr double      kVmax         = 50.0;     // clamp velocity magnitude
    constexpr std::size_t kIterationMax = 10'000UZ; // total iterations
    constexpr std::size_t dontRepelFor  = 10UZ;     // number of iterations until the blocks roughly ordered themselves into rows and columns and then need to be repelled
    // N.B. edges should also already have a rough repulsive force for interconnected blocks

    [[maybe_unused]] constexpr bool onlyWhenClose = true; // true = guard near contact, false = FR-wide
    [[maybe_unused]] constexpr bool capWhenOK     = true; // stop pushing once at/over nominal
    constexpr double                eps           = 1e-6; // numeric guard

    auto needXY = [&](const std::shared_ptr<gr::BlockModel>& A, const std::shared_ptr<gr::BlockModel>& B) -> Point<T> { //
        return minRequiredGapDistance<T>(A, B, cfg) + (size<T>(A) + 1UZ) / 2UZ + (size<T>(B) + 1UZ) / 2UZ;
    };

    // how much we are *short* of the nominal clearance (per axis)
    [[maybe_unused]] auto missXY = [&](const std::shared_ptr<gr::BlockModel>& A, const std::shared_ptr<gr::BlockModel>& B) -> Point<T> {
        const Point<T> need = needXY(A, B);
        const Point<T> diff = centre<T>(A) - centre<T>(B);
        return {std::max<T>(0, need.x - std::abs(diff.x)), std::max<T>(0, need.y - std::abs(diff.y))};
    };

    auto clampVelocity = [&](Point<T>& velocity_) {
        const double s = std::hypot(velocity_.x, velocity_.y);
        if (s > kVmax && s > eps) {
            velocity_ *= (kVmax / s);
        }
    };

    auto wantFor = [&](const std::shared_ptr<gr::BlockModel>& a, const std::shared_ptr<gr::BlockModel>& b) -> double {
        const auto need = minRequiredGapDistance<double>(a, b, cfg);
        return std::max<double>(eps, std::hypot(need.x, need.y));
    };
    auto uNorm = [&](double r, double want) -> double {
        return std::max<double>(eps, r / std::max<double>(eps, want)); // u<1 => inside shell
    };

    std::size_t           iteration;
    std::vector<Point<T>> force(N); // force vector on block
    double                vmaxSeen = 0.0;
    for (iteration = 0UZ; iteration < kIterationMax; ++iteration) {
        vmaxSeen = 0.0;
        std::ranges::fill(force, Point<T>{T(0), T(0)}); // initialise forces
        // refresh anchors positions
        for (size_t i = 0; i < N; ++i) {
            updateBlock(graph.blocks()[i], position[i]);
        }
        // get right-bottom-most edge
        Point<T> canvasSize = {0, 0};
        for (const std::shared_ptr<gr::BlockModel>& node : graph.blocks()) {
            auto bottomRightPos = bottomRight<T>(node);
            canvasSize.x        = std::max(canvasSize.x, bottomRightPos.x);
            canvasSize.y        = std::max(canvasSize.y, bottomRightPos.y);
        }

        // Per-iteration gains/common helpers
        const double repelGain = (dontRepelFor == 0) ? cfg.kRepel : (iteration >= dontRepelFor ? cfg.kRepel : cfg.kRepel * (double(iteration) / double(std::max<std::size_t>(1, dontRepelFor))));

        // --- Repulsive force (returns vector along (i->j)) --------------------
        const auto repel = [&](const std::shared_ptr<gr::BlockModel>& A, const std::shared_ptr<gr::BlockModel>& B) -> Point<double> {
            using enum LayoutAlgorithm;
            const Point<T> diff = centre<double>(B) - centre<double>(A);
            const double   r    = std::max(std::hypot(diff.x, diff.y), eps);
            const Point<T> unit = diff / r;

            const Point<T> minBlockDistance = minRequiredGapDistance<double>(A, B, cfg) + size<double>(A) / 2.0 + size<double>(B) / 2.0;

            switch (cfg.algorithm) {
            case FruchtermanReingold: { // f_rep = l^2 * vec(v,u) / ||v-u||
                const Point<T> minGap = minRequiredGapDistance<double>(A, B, cfg);
                double         l      = std::max({minGap.x, minGap.y, size<double>(A).x, size<double>(B).x, size<double>(A).y, size<double>(B).y}); // largest dimension in x or y
                return cfg.kFruchtermanReingoldScale * l * l / r * unit;
            }
            case EadesLog: { // f_rep = c / r^N
                const double fmag = repelGain / std::pow(r, cfg.kCoulombPower);
                return (fmag / r) * diff;
            }
            case ManhattanPerAxis: { // axis-minimal push only if illegal
                const auto   ad   = Point<double>{std::abs(diff.x), std::abs(diff.y)};
                const double penX = minBlockDistance.x - ad.x;
                const double penY = minBlockDistance.y - ad.y;
                if (penX <= 0.0 && penY <= 0.0) {
                    return {0.0, 0.0};
                }
                constexpr double slip = 0.25;
                Point<double>    f{0.0, 0.0};
                if (penX <= penY) {
                    const double sx = diff.x >= 0.0 ? +1.0 : -1.0;
                    const double g  = repelGain * (0.5 + penX / std::max(minBlockDistance.x, 1.0));
                    f.x             = g * sx * penX;
                    f.y             = g * slip * (diff.y >= 0.0 ? +1.0 : -1.0) * std::min(penX, std::max(0.0, penY));
                } else {
                    const double sy = diff.y >= 0.0 ? +1.0 : -1.0;
                    const double g  = repelGain * (0.5 + penY / std::max(minBlockDistance.y, 1.0));
                    f.y             = g * sy * penY;
                    f.x             = g * slip * (diff.x >= 0.0 ? +1.0 : -1.0) * std::min(penY, std::max(0.0, penX));
                }
                return f;
            }
            case NormalizedRadial:
            default: {                          // your normalized Coulomb in u = r/want
                constexpr std::size_t P    = 2; // 1/u or 1/u^2
                const double          want = wantFor(A, B);
                const double          u    = uNorm(r, want);
                if (u >= 1.0 && capWhenOK) {
                    return {0.0, 0.0};
                }
                const double inv  = (P == 1 ? (1.0 / u) : (1.0 / (u * u)));
                const double fmag = repelGain * inv;
                return (fmag / r) * diff;
            }
            }
        };

        // --- Attractive force (returns vector along (A->B)) -------------------
        [[maybe_unused]] const auto attract = [&](const std::shared_ptr<gr::BlockModel>& A, const std::shared_ptr<gr::BlockModel>& B, const Point<double>& d, double r, double want) -> Point<double> {
            using enum LayoutAlgorithm;
            switch (cfg.algorithm) {
            case FruchtermanReingold: { // f_attr = (r_eff^2 / k) * û
                const Point<T> unit   = d / r;
                const Point<T> minGap = minRequiredGapDistance<double>(A, B, cfg);
                double         l      = std::max({minGap.x, minGap.y, size<double>(A).x, size<double>(B).x, size<double>(A).y, size<double>(B).y}); // largest dimension in x or y
                return cfg.kFruchtermanReingoldScale * r * r / l * unit;
                const double rEff = std::max(eps, r - l);                        // distance minus node extents
                return (rEff * rEff) / cfg.kFruchtermanReingoldScale * l * unit; // attractive force pulls A toward B
            }
            case EadesLog: { // f_attr(u,v) = kSpring * log(r / want) - f_repl(u,v)
                return cfg.kSpring * std::log(std::max<double>(eps, r / want)) * (d / r) - repel(A, B);
            }
            case ManhattanPerAxis: { // dead-zone Hooke per axis
                const auto gap = minRequiredGapDistance<double>(A, B, cfg);
                auto       dz  = [](double delta, double g) {
                    const double a = std::abs(delta);
                    if (a <= g) {
                        return 0.0;
                    }
                    return (delta >= 0.0 ? (a - g) : -(a - g));
                };
                return {cfg.kSpring * dz(d.x, gap.x), cfg.kSpring * dz(d.y, gap.y)};
            }
            case NormalizedRadial:
            default: { // your normalized Hooke: (r - want)/want
                const double ext  = (r - want) / want;
                const double fmag = cfg.kSpring * ext;
                return (fmag / r) * d;
            }
            }
        };

        //  (A) Springs for every edge (use anchorPoint())
        for (auto const& edge : graph.edges()) {
            const std::size_t u = *gr::graph::blockIndex(graph, edge.sourceBlock());
            const std::size_t v = *gr::graph::blockIndex(graph, edge.destinationBlock());

            const std::shared_ptr<gr::BlockModel>& A       = graph.blocks()[u];
            const std::shared_ptr<gr::BlockModel>& B       = graph.blocks()[v];
            PortInfo                               srcPort = getPortByDefinition<gr::PortDirection::OUTPUT>(A, edge.sourcePortDefinition(), loc);
            PortInfo                               dstPort = getPortByDefinition<gr::PortDirection::INPUT>(B, edge.destinationPortDefinition(), loc);
            const Point<double>                    from    = srcPort.anchorPoint<double>();
            const Point<double>                    to      = dstPort.anchorPoint<double>();
            const Point<double>                    d       = to - from;
            double                                 r       = std::hypot(d.x, d.y);
            if (r < eps) {
                r = eps;
            }

            const double        want      = wantFor(A, B);
            const Point<double> edgeForce = attract(A, B, d, r, want);

            force[u] += edgeForce;
            force[v] -= edgeForce;
        }

        // ─────────────────────────────────────────────────────────────
        // (B) Repulsion between nodes (centers), algorithm-specific
        // ─────────────────────────────────────────────────────────────
        [[maybe_unused]] constexpr std::size_t P    = 2;   // 1 => 1/u, 2 => 1/u^2 in normalized modes
        [[maybe_unused]] constexpr double      uFar = 2.5; // optional fade range for normalized modes

        [[maybe_unused]] auto fade = [](double a, double b, double u) {
            if (u <= a) {
                return 1.0;
            }
            if (u >= b) {
                return 0.0;
            }
            const double t = (u - a) / (b - a);
            return 1.0 - (t * t * (3.0 - 2.0 * t)); // smoothstep
        };

        for (std::size_t i = 0; i < N; ++i) {
            const auto ci = centre<double>(graph.blocks()[i]);

            for (std::size_t j = i + 1; j < N; ++j) {
                const auto          cj   = centre<double>(graph.blocks()[j]);
                const Point<double> diff = cj - ci;
                double              r    = std::hypot(diff.x, diff.y);
                if (r < eps) {
                    r = eps;
                }

                const Point<double> f = repel(graph.blocks()[i], graph.blocks()[j]);
                force[i] -= f;
                force[j] += f;
            }
        }

        // (C) gravity-like force
        if (cfg.layout == LayoutPref::HORIZONTAL) {
            // for (size_t i = 0; i < N; ++i) {
            //     if (nodes[i].inputCount() == 0UZ) { // gently nudge sources towards the top-left
            //         force[i].x -= kSrcTether;       // anchor sources (zero inputs) to the left border
            //         // force[i].y += -kSrcTether * effectiveWeights[i]; // having a ballon-like gravity effect depending on children
            //     } else if (nodes[i].outputCount() == 0UZ) { // gently nudge sink towards the top-right
            //         force[i].x += kSnkTether;
            //         // force[i].y += -kSnkTether * effectiveWeights[i];               // having a ballon-like gravity effect depending on children
            //     } else { // all other nodes
            //         // force[i].x += +gravRight / std::max(eps, effectiveWeights[i]); // sort-of sorts nodes based on how many connected children they have
            //         // force[i].y += -gravUp * effectiveWeights[i];
            //     }
            //     force[i].y += -gravUp * effectiveWeights[i];
            // }
        } else {
            // mirror X<->Y for vertical layouts
            for (size_t i = 0; i < N; ++i) {
                force[i].y += gravRight * effectiveWeights[i];
                force[i].x += -gravUp * effectiveWeights[i];
                force[i].y += -gravLeftW;

                if (inputCount(graph.blocks()[i]) == 0) {
                    const double kSrcWall = 2.5;
                    const double minX     = double(std::max(cfg.minMargin, cfg.minGap + cfg.minPortDistance * portsOnSide(graph.blocks()[i], Side::Left)));
                    const double minY     = double(std::max(cfg.minMargin, cfg.minGap + cfg.minPortDistance * portsOnSide(graph.blocks()[i], Side::Top)));
                    force[i].y += -kSrcWall * std::max(0.0, position[i].y - minY);
                    force[i].x += -kSrcWall * std::max(0.0, position[i].x - minX);
                }
            }
        }

        // (D) integrate
        if (true) {
            // Euler based method
            for (std::size_t i = 0UZ; i < N; ++i) {
                const T mass = effectiveWeights[i]; // usually '1' (consider using "large" block equivalent to scale this)
                velocity[i]  = (velocity[i] + force[i] / mass * kStep) * kDampening;

                clampVelocity(velocity[i]);
                vmaxSeen = std::max(vmaxSeen, std::hypot(velocity[i].x, velocity[i].y));
                position[i] += velocity[i] * kStep;
            }
        } else {

            for (std::size_t i = 0UZ; i < N; ++i) {
                const T  mass = effectiveWeights[i];           // usually '1' (consider using "large" block equivalent to scale this)
                Point<T> a    = force[i] * (1.0 / mass);       // acceleration F = ma
                Point<T> v    = position[i] - prevPosition[i]; // implicit velocity - N.B. already expressed in units of 'step' thus no v*step further down
                vmaxSeen      = std::max(vmaxSeen, std::hypot(v.x, v.y));

                prevPosition[i] = position[i];
                position[i]     = position[i] + v * (1.0 - kDrag) + a * (kStep * kStep);
            }
        }

        // optionally optimise port sides to shorten springs
        if (vmaxSeen < 0.01 && iteration >= dontRepelFor) {
            [[maybe_unused]] std::size_t nOptimised = optimiseSideFlips<T>(graph, /*maxIters=*/1, /*improveEps=*/0.00);
        }

        if (iteration >= dontRepelFor && vmaxSeen < 0.001) {
            break;
        }
    } // for (iteration = 0UZ; iteration < kIterationMax; ... ) [..]

    // 5) final snap + update anchors
    for (std::size_t i = 0UZ; i < N; ++i) {
        // nodes[i].update(std::floor(position[i].x), std::floor(position[i].y));
        updateBlock(graph.blocks()[i], position[i]);
    }

    // post-fix pusb top-left node into view
    Point<double> topLeftPos{};
    std::ranges::for_each(graph.blocks(), [&topLeftPos](const std::shared_ptr<gr::BlockModel>& node) { topLeftPos = min(topLeftPos, topLeft<double>(node)); });
    Point<double> diff = topLeftPos - Point<double>(static_cast<double>(cfg.minMargin), static_cast<double>(cfg.minMargin));
    Point<double> offset{diff.x < 0.0 ? std::floor(-diff.x) + 1.0 : 0.0, diff.y < 0.0 ? std::floor(-diff.y) + 1.0 : 0.0};
    std::ranges::for_each(graph.blocks(), [&offset](std::shared_ptr<gr::BlockModel>& node) { updateBlock(node, topLeft<double>(node) + offset); });
}

template<typename T = double>
static void layoutSugiyama(gr::Graph& graph, const LayoutPreference& config = {}, std::source_location loc = std::source_location::current()) {
    if (graph.blocks().empty()) {
        return;
    }

    const std::size_t N = graph.blocks().size();
    for (std::shared_ptr<gr::BlockModel>& n : graph.blocks()) {
        if (getLayoutPref(n) == LayoutPref::UNDEFINED) {
            setLayoutPref(n, config.layout == LayoutPref::AUTO ? LayoutPref::HORIZONTAL : config.layout);
        }
        updateBlock(n, getPosition<T>(n) == Point<T>::undefined() ? Point<T>{static_cast<T>(config.minMargin), static_cast<T>(config.minMargin)} : getPosition<T>(n));
    }

    // ─────────────────────────────────────────────────
    // 1) Adjacency + cycle-breaking
    // ─────────────────────────────────────────────────
    // std::vector<std::vector<std::size_t>> out(N), in(N);
    std::map<std::shared_ptr<gr::BlockModel>, std::vector<std::shared_ptr<gr::BlockModel>>> out{};
    std::map<std::shared_ptr<gr::BlockModel>, std::vector<std::shared_ptr<gr::BlockModel>>> in{};
    for (const auto& e : graph.edges()) {
        if (e.sourceBlock() != e.destinationBlock()) {
            out[e.sourceBlock()].push_back(e.destinationBlock());
            in[e.destinationBlock()].push_back(e.sourceBlock());
        }
    }

    std::ranges::for_each(graph.edges(), [](gr::Edge& e) { setEdgeType(e, EdgeType::Forward); }); // will refine to Lateral after DFS
    std::vector<gr::graph::FeedbackLoop> feedbackLoops = gr::graph::detectFeedbackLoops(graph);
    for (auto& loop : feedbackLoops) {
        setEdgeType(loop.edges.back(), EdgeType::Feedback);
    }
    // alternative use the Eades-Lin-Smyth heuristic for cycle breaking
    // P. Eades, X. Lin, and w.F. Smyth. A fast and effective heuristic for the feedback arc set problem. Information Processing Letters, 47(6): 319-323,1993

    // helper functions for traversing the DAG with reversed edges
    auto predecessors = [&](std::size_t v) -> std::vector<std::size_t> {
        std::vector<std::size_t> preds;
        for (const auto& e : graph.edges()) {
            std::size_t srcIdx = *gr::graph::blockIndex(graph, e.sourceBlock());
            std::size_t dstIdx = *gr::graph::blockIndex(graph, e.destinationBlock());
            if (dstIdx == v && getEdgeType(e) == EdgeType::Forward) {
                preds.push_back(srcIdx);
            } else if (srcIdx == v && getEdgeType(e) == EdgeType::Feedback) {
                preds.push_back(dstIdx); // reversed edge
            }
        }
        return preds;
    };

    auto successors = [&](std::size_t v) -> std::vector<std::size_t> {
        std::vector<std::size_t> succs;
        for (const auto& e : graph.edges()) {
            std::size_t srcIdx = *gr::graph::blockIndex(graph, e.sourceBlock());
            std::size_t dstIdx = *gr::graph::blockIndex(graph, e.destinationBlock());
            if (srcIdx == v && getEdgeType(e) == EdgeType::Forward) {
                succs.push_back(dstIdx);
            } else if (dstIdx == v && getEdgeType(e) == EdgeType::Feedback) {
                succs.push_back(srcIdx); // reversed edge
            }
        }
        return succs;
    };

    // Calculate maxDFS per component with component-aware offsets
    std::vector<std::size_t> maxDFS(N, gr::meta::invalid_index);
    std::size_t              col_offset = 0;

    // Calculate maxDFS only within this component
    std::function<void(std::size_t, std::size_t)> dfs = [&](std::size_t v, std::size_t dist) {
        maxDFS[v] = std::min(maxDFS[v], dist + col_offset);

        for (auto u : successors(v)) {
            dfs(u, dist + 1UZ);
        }
    };

    // find sources in this component
    for (std::size_t v = 0UZ; v < graph.blocks().size(); ++v) {
        bool is_source = true;
        for ([[maybe_unused]] auto p : predecessors(v)) {
            is_source = false;
            break;
        }
        if (is_source) {
            dfs(v, 0UZ);
        }
    }

    // Find max column used by this component
    std::size_t max_col = 0;
    for (std::size_t v = 0UZ; v < graph.blocks().size(); ++v) {
        if (maxDFS[v] != gr::meta::invalid_index) {
            max_col = std::max(max_col, maxDFS[v] - col_offset);
        }
    }

    // Update offset for next component (with a gap)
    col_offset += max_col + 2; // Add spacing between components

    // Start DFS from all sources
    for (std::size_t v = 0UZ; v < N; ++v) {
        if (predecessors(v).empty()) {
            dfs(v, 0UZ);
        }
    }

    // Classify lateral edges based on DFS distances
    for (auto& e : graph.edges()) {
        std::size_t srcIdx = *gr::graph::blockIndex(graph, e.sourceBlock());
        std::size_t dstIdx = *gr::graph::blockIndex(graph, e.destinationBlock());
        if (getEdgeType(e) == EdgeType::Forward          //
            && maxDFS[srcIdx] != gr::meta::invalid_index //
            && maxDFS[dstIdx] != gr::meta::invalid_index //
            && maxDFS[srcIdx] == maxDFS[dstIdx]) {
            setEdgeType(e, EdgeType::Lateral);
        }
    }

    // Post-fix: detect micro-loops and reassign intermediate nodes
    for (const auto& e : graph.edges()) {
        if (getEdgeType(e) != EdgeType::Lateral) {
            continue;
        }
        std::size_t srcIdx = *gr::graph::blockIndex(graph, e.sourceBlock());
        std::size_t dstIdx = *gr::graph::blockIndex(graph, e.destinationBlock());
        // Find nodes that bridge this lateral edge
        for (std::size_t v = 0; v < N; ++v) {
            if (v == srcIdx || v == dstIdx) {
                continue;
            }

            // Check if v is on a path from srcNode to dstNode
            bool hasPathFromSrc = false;
            bool hasPathToDst   = false;

            for (const auto& e2 : graph.edges()) {
                std::size_t srcIdx2 = *gr::graph::blockIndex(graph, e2.sourceBlock());
                std::size_t dstIdx2 = *gr::graph::blockIndex(graph, e2.destinationBlock());
                if (srcIdx2 == srcIdx && dstIdx2 == v) {
                    hasPathFromSrc = true;
                }
                if (srcIdx2 == v && dstIdx2 == dstIdx) {
                    hasPathToDst = true;
                }
            }

            if (hasPathFromSrc && hasPathToDst && maxDFS[v] != maxDFS[srcIdx]) {
                maxDFS[v] = maxDFS[srcIdx];
            }
        }
    }

    // Determine number of columns needed
    std::size_t num_cols = *std::ranges::max_element(maxDFS) + 1UZ;

    // ─────────────────────────────────────────────────
    // 2b) Place nodes in grid based on maxDFS
    // ─────────────────────────────────────────────────

    // Grid: [column][row] -> node_id (or invalid_index if empty)
    std::vector<std::vector<std::size_t>> grid(num_cols);
    std::vector<Point<std::size_t>>       grid_pos(N); // (col, row) for each node

    // Sort nodes by maxDFS, then by node index for stable ordering
    std::vector<std::size_t> node_order(N);
    std::iota(node_order.begin(), node_order.end(), 0);
    std::ranges::stable_sort(node_order, [&](std::size_t a, std::size_t b) {
        if (maxDFS[a] != maxDFS[b]) {
            return maxDFS[a] < maxDFS[b];
        }
        return a < b;
    });

    // Place each node in the first available row of its column
    std::vector<std::size_t> next_free_row(num_cols, 0);

    std::size_t maxRow = 0UZ;
    for (std::size_t v : node_order) {
        std::size_t col = maxDFS[v];
        std::size_t row = next_free_row[col]++;

        if (grid[col].size() <= row) {
            grid[col].resize(row + 1, gr::meta::invalid_index);
        }
        maxRow = std::max(maxRow, row);
    }

    // resize grid to have the same number of max rows for each column
    maxRow += 1UZ;
    for (auto& col : grid) {
        col.resize(maxRow, gr::meta::invalid_index);
    }

    std::ranges::fill(next_free_row, 0UZ);
    for (std::size_t nodeID = 0UZ; nodeID < graph.blocks().size(); ++nodeID) {
        std::size_t col  = maxDFS[nodeID];
        std::size_t row  = next_free_row[col]++;
        grid[col][row]   = nodeID;
        grid_pos[nodeID] = {col, row};
    }

    // ─────────────────────────────────────────────────
    // 3) Crossing minimization (enhanced with edge length penalty)
    // ─────────────────────────────────────────────────

    // Small helpers used below
    auto indeg = [&](std::size_t v) {
        std::size_t d = 0;
        for (auto&& _ : predecessors(v)) {
            (void)_, ++d;
        }
        return d;
    };
    auto outdeg = [&](std::size_t v) {
        std::size_t d = 0;
        for (auto&& _ : successors(v)) {
            (void)_, ++d;
        }
        return d;
    };

    // Prefer predecessor closest in column index (ideally previous column)
    auto main_pred = [&](std::size_t v, std::size_t col) -> std::size_t {
        std::size_t best = gr::meta::invalid_index, best_cdist = std::numeric_limits<std::size_t>::max();
        for (auto p : predecessors(v)) {
            std::size_t c     = grid_pos[p].x;
            std::size_t cdist = (c > col) ? (c - col) : (col - c);
            if (cdist < best_cdist) {
                best_cdist = cdist;
                best       = p;
            }
        }
        return best;
    };
    // Pull strength to keep short chains straight
    auto straight_bias = [&](std::size_t v, std::size_t p) -> double {
        if (p == gr::meta::invalid_index) {
            return 0.0;
        }
        double w = 6.0; // base
        if (outdeg(p) == 1) {
            w *= 1.8; // predecessor has single fan-out
        }
        if (outdeg(v) == 0 && indeg(v) == 1) {
            w *= 1.5; // v looks like a sink in a chain
        }
        return w;
    };

    // Evaluate cost for assigning `nodes[i]` to row `rows[i]` in column `col`
    auto eval_cost = [&](std::size_t col, const std::vector<std::size_t>& nodes_, const std::vector<std::size_t>& rows) -> double {
        // Save and place temporarily
        std::vector<Point<std::size_t>> saved;
        saved.reserve(nodes_.size());
        for (auto v : nodes_) {
            saved.push_back(grid_pos[v]);
        }
        for (std::size_t i = 0UZ; i < nodes_.size(); ++i) {
            grid_pos[nodes_[i]] = {col, rows[i]};
        }

        auto restore = [&]() {
            for (std::size_t i = 0UZ; i < nodes_.size(); ++i) {
                grid_pos[nodes_[i]] = saved[i];
            }
        };

        double cost = 0.0;

        // crossings with previous & next columns (heavy penalty)
        for (std::size_t i = 0UZ; i < nodes_.size(); ++i) {
            for (std::size_t j = i + 1; j < nodes_.size(); ++j) {
                auto a = nodes_[i], b = nodes_[j];

                if (col > 0UZ) {
                    for (auto pa : predecessors(a)) {
                        if (grid_pos[pa].x == col - 1) {
                            for (auto pb : predecessors(b)) {
                                if (grid_pos[pb].x == col - 1) {
                                    if (grid_pos[pa].y > grid_pos[pb].y) {
                                        cost += 10.0;
                                    }
                                }
                            }
                        }
                    }
                }
                if (col + 1 < grid.size()) {
                    for (auto sa : successors(a)) {
                        if (grid_pos[sa].x == col + 1) {
                            for (auto sb : successors(b)) {
                                if (grid_pos[sb].x == col + 1) {
                                    if (grid_pos[sa].y > grid_pos[sb].y) {
                                        cost += 10.0;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Edge length penalties (closer columns weigh more)
        auto add_lengths = [&](std::size_t v) {
            const std::size_t vr = grid_pos[v].y;
            for (const auto& e : graph.edges()) {
                std::size_t srcIdx = *gr::graph::blockIndex(graph, e.sourceBlock());
                std::size_t dstIdx = *gr::graph::blockIndex(graph, e.destinationBlock());
                double      weight = 1.0;

                // Adjust weight based on edge type
                if (getEdgeType(e) == EdgeType::Feedback) {
                    weight *= 0.5; // Less important to minimize feedback edges
                } else if (getEdgeType(e) == EdgeType::Lateral) {
                    weight *= 2.0; // More important to keep lateral edges short
                }

                if (srcIdx == v) {
                    std::size_t u     = dstIdx;
                    std::size_t cdist = (grid_pos[u].x > col) ? (grid_pos[u].x - col) : (col - grid_pos[u].x);
                    if (cdist > 0) {
                        weight /= double(cdist);
                        cost += weight * double((grid_pos[u].y > vr) ? (grid_pos[u].y - vr) : (vr - grid_pos[u].y));
                    }
                }
                if (dstIdx == v) {
                    std::size_t u     = srcIdx;
                    std::size_t cdist = (grid_pos[u].x > col) ? (grid_pos[u].x - col) : (col - grid_pos[u].x);
                    if (cdist > 0) {
                        weight /= double(cdist);
                        cost += weight * double((grid_pos[u].y > vr) ? (grid_pos[u].y - vr) : (vr - grid_pos[u].y));
                    }
                }
            }
        };
        for (auto v : nodes_) {
            add_lengths(v);
        }

        // Chain-straightening bias (pull node toward main predecessor's row)
        for (auto v : nodes_) {
            auto p = main_pred(v, col);
            if (p == gr::meta::invalid_index) {
                continue;
            }
            const std::size_t want = grid_pos[p].y;
            const std::size_t got  = grid_pos[v].y;
            std::size_t       drow = (want > got) ? (want - got) : (got - want);
            cost += straight_bias(v, p) * double(drow);
        }

        // Bonus for keeping same-component nodes together vertically
        for (auto v : nodes_) {
            std::size_t same_comp_count  = 0;
            std::size_t row_distance_sum = 0;

            // Check all other nodes in this column that belong to same component
            for (auto u : nodes_) {
                if (u != v) {
                    same_comp_count++;
                    std::size_t v_row = grid_pos[v].y;
                    std::size_t u_row = grid_pos[u].y;
                    row_distance_sum += (v_row > u_row) ? (v_row - u_row) : (u_row - v_row);
                }
            }

            if (same_comp_count > 0) {
                // Penalize spreading same-component nodes across rows
                cost += 0.5 * double(row_distance_sum) / double(same_comp_count);
            }
        }

        restore();
        return cost;
    };

    // Utility: next k-combination of {0..n-1} in lexicographic order
    auto next_combination = [](std::vector<std::size_t>& comb, std::size_t n) -> bool {
        const std::size_t k = comb.size();
        if (k == 0) {
            return false;
        }
        for (std::size_t i = k; i-- > 0;) {
            if (comb[i] != i + (n - k)) {
                ++comb[i];
                for (std::size_t j = i + 1; j < k; ++j) {
                    comb[j] = comb[j - 1] + 1;
                }
                return true;
            }
        }
        return false;
    };

    for (int iter = 0; iter < 10; ++iter) {
        bool changed = false;

        for (std::size_t col = 0; col < grid.size(); ++col) {
            // Collect nodes present in this column
            std::vector<std::size_t> nodesInCol;
            nodesInCol.reserve(grid[col].size());
            for (std::size_t r = 0; r < grid[col].size(); ++r) {
                if (grid[col][r] != gr::meta::invalid_index) {
                    nodesInCol.push_back(grid[col][r]);
                }
            }

            const std::size_t k = nodesInCol.size();
            if (k < 2UZ) {
                continue;
            }

            // --- Small columns: exact search over row-combinations × permutations ---
            if (k <= 6 && maxRow <= 10) {
                // Start with the first k rows as the initial combination
                std::vector<std::size_t> rows(k);
                std::iota(rows.begin(), rows.end(), 0);

                // We'll search all k-combinations of rows and all permutations of nodes
                std::vector<std::size_t> best_rows  = rows;
                std::vector<std::size_t> best_nodes = nodesInCol;
                double                   best_cost  = std::numeric_limits<double>::infinity();

                std::vector<std::size_t> nodesPerm = nodesInCol;
                do {
                    // reset rows to the first combination for each new permutation
                    rows.resize(k);
                    std::iota(rows.begin(), rows.end(), 0);
                    do {
                        double c = eval_cost(col, nodesPerm, rows);
                        if (c < best_cost) {
                            best_cost  = c;
                            best_rows  = rows;
                            best_nodes = nodesPerm;
                        }
                    } while (next_combination(rows, maxRow));
                } while (std::ranges::next_permutation(nodesPerm).found);

                // Apply: clear column, place nodes at chosen rows
                for (std::size_t r = 0; r < maxRow; ++r) {
                    if (grid[col][r] != gr::meta::invalid_index) {
                        grid[col][r] = gr::meta::invalid_index;
                    }
                }
                for (std::size_t i = 0; i < k; ++i) {
                    const std::size_t v = best_nodes[i];
                    const std::size_t r = best_rows[i];
                    if (grid_pos[v] != Point<std::size_t>{col, r}) {
                        changed = true;
                    }
                    grid[col][r] = v;
                    grid_pos[v]  = {col, r};
                }
            }
            // --- Large columns: weighted barycenter + nearest-free-row assignment ---
            else {
                struct Score {
                    double      pos;
                    std::size_t v;
                };
                std::vector<Score> scores;
                scores.reserve(k);

                for (auto v : nodesInCol) {
                    double sum = 0.0, wsum = 0.0;

                    for (auto p : predecessors(v)) {
                        double w = 1.0 / (1.0 + std::abs(static_cast<double>(grid_pos[p].y) - static_cast<double>(col)));
                        sum += w * static_cast<double>(grid_pos[p].y);
                        wsum += w;
                    }
                    for (auto s : successors(v)) {
                        double w = 1.0 / (1.0 + std::abs(static_cast<double>(grid_pos[s].y) - static_cast<double>(col)));
                        sum += w * static_cast<double>(grid_pos[s].y);
                        wsum += w;
                    }

                    double pos = (wsum > 0.0) ? (sum / wsum) : static_cast<double>(grid_pos[v].y);

                    if (auto p = main_pred(v, col); p != gr::meta::invalid_index) {
                        double w = straight_bias(v, p), want = static_cast<double>(grid_pos[p].y);
                        pos = (pos + w * want) / (1.0 + w);
                    }
                    scores.push_back({pos, v});
                }

                // sort by desired position and assign the nearest free rows
                std::ranges::stable_sort(scores, [](const Score& a, const Score& b) { return a.pos < b.pos; });

                std::vector<int> freeRows(maxRow);
                std::iota(freeRows.begin(), freeRows.end(), 0);
                // clear column before re-fill
                for (std::size_t r = 0; r < maxRow; ++r) {
                    if (grid[col][r] != gr::meta::invalid_index) {
                        grid[col][r] = gr::meta::invalid_index;
                    }
                }

                auto take_nearest = [&](double target) -> int {
                    const auto it     = std::ranges::lower_bound(freeRows, static_cast<int>(std::round(target)));
                    int        cand1  = (it == freeRows.end() ? freeRows.back() : *it);
                    int        cand0  = (it == freeRows.begin() ? cand1 : *(it - 1));
                    int        chosen = (std::abs(static_cast<double>(cand0) - target) <= std::abs(static_cast<double>(cand1) - target)) ? cand0 : cand1;
                    freeRows.erase(std::ranges::find(freeRows, chosen));
                    return chosen;
                };

                for (auto sc : scores) {
                    int r = take_nearest(sc.pos);
                    if (grid_pos[sc.v] != Point<std::size_t>{col, static_cast<std::size_t>(r)}) {
                        changed = true;
                    }
                    grid[col][static_cast<std::size_t>(r)] = sc.v;
                    grid_pos[sc.v]                         = {col, static_cast<std::size_t>(r)};
                }
            }
        }

        if (!changed) {
            break;
        }
    }

    // ─────────────────────────────────────────────────
    // 4) Calculate cell dimensions with routing gaps
    // ─────────────────────────────────────────────────
    struct cell_info {
        T paddingBefore{0};
        T size{0};
        T paddingAfter{0};
    };

    const std::size_t      max_row = std::ranges::max(grid | std::views::transform(&std::vector<std::size_t>::size));
    std::vector<cell_info> colWidth(num_cols + 1UZ); // includes one additional empty column at the end
    std::vector<cell_info> rowHeight(max_row + 2UZ); // includes one additional empty row at the end
    auto                   computeMaxCellDimensions = [&] {
        // find max dimensions per column and row, including gaps for routing

        for (std::size_t col = 0; col < num_cols; ++col) {
            for (std::size_t row = 0; row < grid[col].size(); ++row) {
                std::size_t nodeIdx = grid[col][row];
                if (nodeIdx == gr::meta::invalid_index) {
                    continue; // skip empty cells/non-drawable nodes
                }
                colWidth[col].paddingBefore = std::max(colWidth[col].paddingBefore, static_cast<T>(portsOnSide(graph.blocks()[nodeIdx], Side::Left) * config.minPortDistance));
                colWidth[col].size          = std::max(colWidth[col].size, size<T>(graph.blocks()[nodeIdx]).x);
                colWidth[col].paddingAfter  = std::max(colWidth[col].paddingAfter, static_cast<T>(portsOnSide(graph.blocks()[nodeIdx], Side::Right) * config.minPortDistance));
            }
        }

        for (std::size_t col = 0UZ; col < num_cols; ++col) {
            for (std::size_t row = 0UZ; row < grid[col].size(); ++row) {
                std::size_t nodeIdx = grid[col][row];
                if (nodeIdx == gr::meta::invalid_index) {
                    continue; // skip empty cells/non-drawable nodes
                }
                auto& node                   = graph.blocks()[nodeIdx];
                rowHeight[row].paddingBefore = std::max(rowHeight[row].paddingBefore, static_cast<T>((portsOnSide(node, Side::Top) > 0 ? 1 : 0) * config.minPortDistance));
                rowHeight[row].size          = std::max(rowHeight[row].size, size<T>(node).y);
                rowHeight[row].paddingAfter  = std::max(rowHeight[row].paddingAfter, static_cast<T>((portsOnSide(node, Side::Bottom) > 0 ? 1 : 0) * config.minPortDistance));
            }
        }
    };

    computeMaxCellDimensions();

    // ─────────────────────────────────────────────────
    // 5) Convert grid to world coordinates
    // ─────────────────────────────────────────────────
    auto physicalScreenLayout = [&] {
        T columnOffset{colWidth[0UZ].paddingBefore};
        for (std::size_t col = 0UZ; col < num_cols; col++) {
            T rowOffset{static_cast<T>(rowHeight[0UZ].paddingBefore)};
            for (std::size_t row = 0UZ; row < grid[col].size(); row++) {
                std::size_t nodeIdx = grid[col][row];
                if (nodeIdx == gr::meta::invalid_index) {
                    continue; // skip empty cells/non-drawable nodes
                }
                updateBlock(graph.blocks()[nodeIdx], Point<T>{columnOffset, rowOffset});
                rowOffset += rowHeight[row].size + std::max(std::max(rowHeight[row].paddingAfter, rowHeight[row + 1UZ].paddingBefore), static_cast<T>(config.minDefaultGap));
            }
            columnOffset += colWidth[col].size + colWidth[col].paddingAfter + std::max(colWidth[col + 1UZ].paddingBefore, static_cast<T>(config.minDefaultGap));
        }
    };

    physicalScreenLayout(); // initial layout

    // ─────────────────────────────────────────────────
    // 5) Port side optimization
    // ─────────────────────────────────────────────────

    if (config.optimiserMaxIterations > 0) {
        optimiseSideFlips<T>(graph, config.optimiserMaxIterations, config.optimiserMinImprovement, loc);
    }

    // ─────────────────────────────────────────────────
    // 6) Post-Fixes
    // ─────────────────────────────────────────────────

    // need to re-layout since changing port anchor positions may change the required gaps between blocks
    computeMaxCellDimensions();
    physicalScreenLayout();
}

//------------------------------------------------------
// Canvas drawing + box characters
//------------------------------------------------------

// helper to flip an incoming movement into the oppositeDirection edge

template<gr::arithmetic_or_complex_like T = std::size_t>
void drawPath(gr::utf8::ImCanvasLike auto& canvas, const gr::Edge& edge, const std::vector<Point<T>>& path, [[maybe_unused]] bool isFeedback = false) {
    const Style       style_ = style(edge);
    const std::size_t n      = path.size();
    if (n < 2) {
        if (n == 1) { // single‐point marker
            canvas.put(path[0], "*", style_);
        }
        return;
    }

    std::vector<Direction> dirs(n, Direction::None);
    for (std::size_t i = 1; i < n; ++i) {
        int dx  = static_cast<int>(path[i].x) - static_cast<int>(path[i - 1].x);
        int dy  = static_cast<int>(path[i].y) - static_cast<int>(path[i - 1].y);
        dirs[i] = gr::utf8::directionFromDelta(dx, dy);
    }

    // draw the first stub at the source cell (outgoing edge only):
    const Point<T>& p0 = path[0UZ];
    if (p0 == Point<T>::undefined()) {
        std::println(stderr, "canvas {}x{} - path for edge '{}' contains undefined coordinates: {}", canvas.width(), canvas.height(), edge, path);
        return;
    }
    canvas.addMask(p0, dirs[1UZ], std::hash<gr::Edge>{}(edge), style_);

    // draw all the internal junctions:
    //    Use the *oppositeDirection* of the incoming bit, OR’d with the outgoing bit.
    for (std::size_t i = 1UZ; i + 1UZ < n; ++i) {
        auto&     p    = path[i];
        Direction mask = oppositeDirection(dirs[i]) // incoming edge
                         | dirs[i + 1UZ];           // outgoing edge
        canvas.addMask(p, mask, std::hash<gr::Edge>{}(edge), style_);
    }

    // draw the final stub at the destination cell (incoming edge only):
    auto& pt = path.back();
    canvas.addMask(pt, oppositeDirection(dirs.back()), std::hash<gr::Edge>{}(edge), style_);
}

//------------------------------------------------------
// Phase 2: materialize & drawing helpers
//------------------------------------------------------
static std::string_view portGlyph(const PortInfo& port, Side side) noexcept {
    using enum Side;

    if (!port.isStream()) {
        return port.isInput() ? "⊖" : "⊕";
    }
    // stream port only
    if (port.isSynchronous()) { // sync port
        switch (side) {
        case Left: return port.isInput() ? "▶" : "◀";
        case Right: return port.isInput() ? "◀" : "▶";
        case Top: return port.isInput() ? "▼" : "▲";
        case Bottom: return port.isInput() ? "▲" : "▼";
        }
    } else { // async port
        switch (side) {
        case Left: return port.isInput() ? "▷" : "◁";
        case Right: return port.isInput() ? "◁" : "▷";
        case Top: return port.isInput() ? "▽" : "△";
        case Bottom: return port.isInput() ? "△" : "▽";
        }
    }
    return "?"; // should never happen
}

template<gr::arithmetic_or_complex_like T = std::size_t>
inline void drawBlock(gr::utf8::ImCanvasLike auto& c, const std::shared_ptr<gr::BlockModel>& block) {
    using enum Side;
    assert(block.get() != nullptr);

    // draw the block's frame
    const Style blockStyle = style(block);
    c.hLine(topLeft<T>(block), topRight<T>(block), blockStyle);
    c.hLine(bottomLeft<T>(block), bottomRight<T>(block), blockStyle);
    c.vLine(topLeft<T>(block), bottomLeft<T>(block), blockStyle);
    c.vLine(topRight<T>(block), bottomRight<T>(block), blockStyle);
    c.blockRect(topLeft<std::size_t>(block), bottomRight<std::size_t>(block));

    // place title
    std::size_t titleSize{gr::utf8::length(block->name())};
    if (portsOnSide(block, Top) == 0UZ && portsOnSide(block, Bottom) == 0UZ) {                    // centred on the top frame
        c.put({centre<T>(block).x - (titleSize >> 1UZ), getPosition<T>(block).y}, block->name()); // no nodes on top or bottom
    } else {                                                                                      // centred inside the box
        c.put({centre<T>(block).x - (titleSize >> 1UZ), centre<T>(block).y}, block->name());
    }

    // draw ports
    auto infos = std::array{std::views::all(getPortInfos<gr::PortDirection::INPUT>(block)), std::views::all(getPortInfos<gr::PortDirection::OUTPUT>(block))};
    for (PortInfo& info : infos | std::views::join) {
        if (!info.preferredSide()) {
            continue;
        }

        Style          style = info.style();
        const Point<T> pt    = info.anchorPoint<T>();
        c.put({pt.x, pt.y}, portGlyph(info, *info.preferredSide()), style);

        switch (*info.preferredSide()) {
        case Left: c.put({pt.x + 1UZ, pt.y}, info.name(), style); break;
        case Right: c.put({pt.x - gr::utf8::length(info.name()), pt.y}, info.name(), style); break;
        case Top: c.put({pt.x + 1UZ, pt.y}, info.name(), style); break;
        case Bottom: c.put({pt.x + 1UZ, pt.y}, info.name(), style); break;
        }
    }
}

//------------------------------------------------------
// Phase 3: Pathfinder via Dijkstra's algorithm
//------------------------------------------------------
template<gr::arithmetic_or_complex_like T = std::size_t>
std::vector<Point<T>> routeDijkstra(const gr::utf8::ImCanvasLike auto& canvas, const gr::Edge& edge, LayoutPreference const& config = {}) {
    using enum Direction;
    constexpr std::array<Direction, 9UZ> idxToMask{{// mask ↔ small index [0..8]
        None, North, East, South, West,             //
        NorthEast, SouthEast, SouthWest, NorthWest}};

    auto toCellIndex = [&](const Point<T>& point) noexcept {
        assert(static_cast<std::size_t>(point.x) < canvas.width());
        assert(static_cast<std::size_t>(point.y) < canvas.height());
        return static_cast<std::size_t>(point.y) * canvas.width() + static_cast<std::size_t>(point.x);
    };

    static constexpr double                                        sqrt2 = std::numbers::sqrt2_v<double>;
    static constexpr std::array<std::tuple<int, int, double>, 8UZ> moves{     // eight possible steps (dx,dy) + their base costs:
        {{1, 0, 1.0}, {-1, 0, 1.0}, {0, 1, 1.0}, {0, -1, 1.0},                // horizontal moves
            {1, 1, sqrt2}, {-1, -1, sqrt2}, {1, -1, sqrt2}, {-1, 1, sqrt2}}}; // diagonal moves

    PortInfo              srcPort   = getPortByDefinition<gr::PortDirection::OUTPUT>(edge.sourceBlock(), edge.sourcePortDefinition());
    PortInfo              dstPort   = getPortByDefinition<gr::PortDirection::INPUT>(edge.destinationBlock(), edge.destinationPortDefinition());
    constexpr std::size_t DIR_COUNT = idxToMask.size();
    std::size_t const     total     = canvas.width() * canvas.height();
    std::size_t const     dstCell   = toCellIndex(dstPort.exitPoint<T>());

    // distance[cell][dirIdx] = best cost so far arriving along that direction
    std::vector<std::array<double, DIR_COUNT>> distance(total);
    for (auto& row : distance) {
        row.fill(std::numeric_limits<double>::infinity());
    }
    std::vector<std::array<Point<T>, DIR_COUNT>>  prevPt(total);
    std::vector<std::array<Direction, DIR_COUNT>> prevDir(total);
    for (auto& row : prevDir) {
        row.fill(None);
    }

    using State = std::tuple<double, Point<T>, Direction, std::size_t>; // cost, (x,y), direction, #steps
    auto cmp    = [](State const& a, State const& b) { return std::get<0>(a) > std::get<0>(b); };

    std::priority_queue<State, std::vector<State>, decltype(cmp)> queue(cmp);

    // prime starting point
    const Point<T>  srcPosition{srcPort.exitPoint<T>()};
    const Direction srcExit{srcPort.exitDir()};
    const Point<T>  dstPosition{dstPort.exitPoint<T>()};
    const Direction dstExit{oppositeDirection(dstPort.exitDir())};
    distance[toCellIndex(srcPosition)][index(srcPort.exitDir())] = 0.0;
    queue.emplace(0.0, srcPosition, srcExit, /*steps*/ 0UZ);

    // Validate positions
    if (srcPosition == Point<T>::undefined() || dstPosition == Point<T>::undefined()) {
        std::println(stderr, "Warning: Invalid port positions for edge {}", edge);
        std::println(stderr, "  Source: {} at {}", srcPort.name(), srcPosition);
        std::println(stderr, "  Dest: {} at {}", dstPort.name(), dstPosition);
        return {}; // Return empty path instead of crashing
    }

    // Dijkstra main loop
    while (!queue.empty()) {
        const auto [curCost, p, currentDirection, steps] = queue.top();
        queue.pop();
        if (const std::size_t cell = toCellIndex(p); curCost > distance[cell][index(currentDirection)]) { // stale entry
            continue;
        }

        // if we reached dst *and* (no dstLock || correct arrival dir)
        if (p == dstPosition && (dstExit == Direction::None || currentDirection == dstExit)) {
            break;
        }

        // expand and explore the eight neighbours
        for (auto const& mv : moves) {
            const auto [dx, dy, baseCost] = mv;
            if ((dx < 0 && p.x == 0) || (dy < 0 && p.y == 0) || (dx > 0 && p.x == canvas.width() - 1UZ) || (dy > 0 && p.y == canvas.height() - 1UZ)) {
                continue; // hitting left, right, top, or bottom border
            }
            Point n{static_cast<std::size_t>(static_cast<int>(p.x) + dx), static_cast<std::size_t>(static_cast<int>(p.y) + dy)};

            const std::size_t nextCell = toCellIndex(n);
            if (canvas.isBlocked(n)) {
                continue; // can't cross node boundaries
            }

            // ban 180° back-reversal edges
            const Direction nextDirection = gr::utf8::directionFromDelta(dx, dy);
            if (currentDirection != None && nextDirection == oppositeDirection(currentDirection)) {
                continue;
            }

            // outgoing lock (hold the initial src direction)
            if (config.outgoingDirectionLock > 0UZ && srcExit != None && steps < config.outgoingDirectionLock && nextDirection != srcExit) {
                continue;
            }

            // incoming lock (hold final dst direction)
            if (config.incomingDirectionLock > 0UZ) {
                if (chebyshevNorm(n, dstPosition) <= config.incomingDirectionLock && nextDirection != dstExit) {
                    continue;
                }
            }

            double turningPenalty = 0.0;
            if (currentDirection != None && nextDirection != currentDirection) {
                int        angle   = abs(angleDegree(currentDirection, nextDirection));
                const bool wasDiag = isDiagonalDirection(currentDirection);
                const bool nowDiag = isDiagonalDirection(nextDirection);
                if (!wasDiag && !nowDiag) { // cardinal axis -> cardinal axis
                    turningPenalty = config.straightPenalty * (angle / 90.0);
                } else if (wasDiag != nowDiag) { // diag <-> cardinal
                    turningPenalty = config.diagonalPenalty * (angle / 45.0);
                } else { // diag -> diag
                    turningPenalty = config.straightPenalty * (angle / 90.0);
                }
            }

            double newCost = curCost + baseCost + baseCost * turningPenalty;
            if (newCost < distance[nextCell][index(nextDirection)]) {
                distance[nextCell][index(nextDirection)] = newCost;
                prevPt[nextCell][index(nextDirection)]   = p;
                prevDir[nextCell][index(nextDirection)]  = currentDirection;
                queue.emplace(newCost, n, nextDirection, steps + 1);
            }
        }
    }

    // ——— 5) choose best arrival at dst —————————————————————————————————
    std::array<double, DIR_COUNT>& arrivals = distance[dstCell];
    std::size_t                    bestIdx  = dstExit == None ? size_t(std::ranges::min_element(arrivals) - arrivals.begin()) : index(dstExit);

    // ——— 6) path reconstruction (walk back via prevPt/prevDir) ————————
    std::vector<Point<T>> path;
    for (Point cur = dstPosition;;) {
        path.push_back(cur);
        if (cur == srcPosition || cur.x >= canvas.width() || cur.y >= canvas.height()) {
            break;
        }
        assert(cur.x < canvas.width());
        assert(cur.y < canvas.height());
        size_t    c  = toCellIndex(cur);
        Direction pd = prevDir[c][bestIdx];
        cur          = prevPt[c][bestIdx];
        bestIdx      = index(pd);
    }
    std::ranges::reverse(path);
    return path;
}

//------------------------------------------------------

template<typename TGraph, gr::arithmetic_or_complex_like T = std::size_t>
requires(std::is_same_v<std::remove_reference_t<TGraph>, gr::Graph>)
[[nodiscard]] static std::string draw(TGraph&& graph, const LayoutPreference& config = {}) {
    std::vector<gr::Graph> graphs = gr::graph::weaklyConnectedComponents(graph);
    if (graphs.size() > 1UZ) {
        std::string      ret;
        LayoutPreference subGraphConfig = config;
        for (auto& g : graphs) {
            ret += draw(std::move(g), subGraphConfig);
            subGraphConfig.minMargin = 0UZ; // ignore margins for subsequent graphs
        }
        return ret;
    }

    // layout/place blocks
    if (config.algorithm == LayoutAlgorithm::Sugiyama) {
        layoutSugiyama(graph, config);
    } else {
        layoutSpringModel(graph, config);
    }

    // compute requried canvas size
    std::size_t W = 0UZ;
    std::size_t H = 0UZ;
    for (const std::shared_ptr<gr::BlockModel>& n : graph.blocks()) {
        Point gap = minRequiredGapDistance<T>(n, n);
        W         = std::max(W, bottomRight<T>(n).x + gap.x);
        H         = std::max(H, bottomRight<T>(n).y + gap.y);
    }
    gr::utf8::ImCanvas<> canvas(W, H);

    // draw block boxes + ports + labels
    for (const std::shared_ptr<gr::BlockModel>& block : graph.blocks()) {
        drawBlock(canvas, block);
    }

    // route edges
    for (const gr::Edge& e : graph.edges()) {
        LayoutPreference routingConfig = config;
        if (getEdgeType(e) == EdgeType::Feedback) {
            routingConfig.straightPenalty *= 0.8; // Allow more turns for feedback
        } else if (getEdgeType(e) == EdgeType::Lateral) {
            routingConfig.straightPenalty *= 1.5; // Prefer straight for lateral
        }

        std::vector<Point<T>> path = routeDijkstra(canvas, e, routingConfig);
        drawPath(canvas, e, path, getEdgeType(e) == EdgeType::Feedback);
    }
    return canvas.toString({}, true);
}

} // namespace gr::graph

#endif // GNURADIO_IMGRAPH_HPP
