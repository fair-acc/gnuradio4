#ifndef GNURADIO_BLOCK_MERGING_HPP
#define GNURADIO_BLOCK_MERGING_HPP

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/Graph.hpp>

namespace gr {

namespace detail {

struct Empty {};

struct PortRef {
    std::shared_ptr<BlockModel> block;
    std::size_t                 portIndex;
};

struct SubBlockPorts {
    std::vector<PortRef> outputPorts;
    std::vector<PortRef> inputPorts;
};

struct GraphWithPortMaps {
    Graph                graph;
    std::vector<PortRef> outputPorts; // external output[i] → leaf PortRef
    std::vector<PortRef> inputPorts;  // external input[i]  → leaf PortRef
};

template<typename T>
concept MergedBlock = requires(const T& t) {
    { t.graphWithPortMaps() } -> std::same_as<GraphWithPortMaps>;
};

template<BlockLike T>
std::shared_ptr<BlockModel> makeNonOwningWrapper(T& block) {
    BlockModel* raw = new BlockWrapper<T, std::false_type>(block);
    return std::shared_ptr<BlockModel>{raw};
}

template<typename T>
SubBlockPorts addSubBlock(Graph& g, T& block) {
    if constexpr (MergedBlock<T>) {
        auto result = block.graphWithPortMaps();
        for (const auto& b : result.graph.blocks()) {
            g.addBlock(b, false);
        }
        for (const auto& e : result.graph.edges()) {
            std::ignore = g.addEdge(e);
        }
        return {std::move(result.outputPorts), std::move(result.inputPorts)};
    } else {
        auto        w    = makeNonOwningWrapper(block);
        std::size_t nOut = traits::block::stream_output_port_types<T>::size;
        std::size_t nIn  = traits::block::stream_input_port_types<T>::size;
        g.addBlock(w, false);
        std::vector<PortRef> outPorts(nOut);
        std::vector<PortRef> inPorts(nIn);
        for (std::size_t i = 0; i < nOut; ++i) {
            outPorts[i] = {w, i};
        }
        for (std::size_t i = 0; i < nIn; ++i) {
            inPorts[i] = {w, i};
        }
        return {std::move(outPorts), std::move(inPorts)};
    }
}

template<typename TDesc>
struct to_left_descriptor : TDesc {
    template<typename TBlock>
    static constexpr decltype(auto) getPortObject(TBlock&& obj) {
        return TDesc::getPortObject(obj._leftBlock);
    }
};

template<typename TDesc>
struct to_right_descriptor : TDesc {
    template<typename TBlock>
    static constexpr decltype(auto) getPortObject(TBlock&& obj) {
        return TDesc::getPortObject(obj._rightBlock);
    }
};

template<typename TDesc>
struct to_forward_descriptor : TDesc {
    template<typename TBlock>
    static constexpr decltype(auto) getPortObject(TBlock&& obj) {
        return TDesc::getPortObject(obj._forward);
    }
};

template<typename B>
void forwardSettings(B& block, const gr::property_map& params) {
    if constexpr (refl::reflectable<std::remove_cvref_t<B>>) {
        if (!params.empty()) {
            std::ignore = block.settings().setStaged(params);
            std::ignore = block.settings().applyStagedParameters();
        }
    }
}

template<typename B>
void forwardNestedSettings(B& block, const gr::property_map& init, std::string_view key) {
    if (auto it = init.find(key); it != init.end()) {
        if (const auto* nested = it->second.template get_if<pmt::Value::Map>()) {
            forwardSettings(block, *nested);
        }
    }
}

template<typename B>
constexpr std::size_t subblockChunkSize() {
    if constexpr (requires {
                      { B::merged_work_chunk_size() } -> std::same_as<std::size_t>;
                  }) {
        return B::merged_work_chunk_size();
    } else {
        return std::dynamic_extent;
    }
}

} // namespace detail

/**
 * @brief Compile-time block merging: fuse processOne chains, bypassing runtime buffers.
 *
 * @code
 * // name-based:  Scale("out") -> Adder("in1")
 * auto m1 = Merge<Scale<2>, "out", Adder<>, "in1">();
 * // index-based: equivalent
 * auto m2 = MergeByIndex<Scale<2>, 0, Adder<>, 0>();
 * @endcode
 */

template<BlockLike Left, std::size_t OutId, //
    BlockLike Right, std::size_t InId>
class MergeByIndex : public Block<MergeByIndex<Left, OutId, Right, InId>> {
    template<typename TDesc>
    friend struct detail::to_right_descriptor;

    template<typename TDesc>
    friend struct detail::to_left_descriptor;

public:
    using OverridePortList = meta::concat<
        // Left:
        typename meta::concat<typename traits::block::all_port_descriptors<Left>::template filter<traits::port::is_message_port>, traits::block::stream_input_ports<Left>, meta::remove_at<OutId, traits::block::stream_output_ports<Left>>>::template transform<detail::to_left_descriptor>,
        // Right:
        typename meta::concat<typename traits::block::all_port_descriptors<Right>::template filter<traits::port::is_message_port>, meta::remove_at<InId, traits::block::stream_input_ports<Right>>, traits::block::stream_output_ports<Right>>::template transform<detail::to_right_descriptor>>;

    using InputPortTypes = typename OverridePortList::template filter<traits::port::is_input_port, traits::port::is_stream_port>::template transform<traits::port::type>;

    using ReturnType = typename OverridePortList::template filter<traits::port::is_output_port, traits::port::is_stream_port>::template transform<traits::port::type>::tuple_or_type;

    GR_MAKE_REFLECTABLE(MergeByIndex);

    gr::meta::immutable<std::string> unique_name = std::format("MergeByIndex<{}:{},{}:{}>#{}", gr::meta::type_name<Left>(), OutId, gr::meta::type_name<Right>(), InId, this->unique_id);

    MergeByIndex(const MergeByIndex& other)       = delete;
    MergeByIndex& operator=(MergeByIndex& other)  = delete;
    MergeByIndex& operator=(MergeByIndex&& other) = delete;

    MergeByIndex(MergeByIndex&& other) noexcept(std::is_nothrow_move_constructible_v<Left> && std::is_nothrow_move_constructible_v<Right>) : _leftBlock(std::move(other._leftBlock)), _rightBlock(std::move(other._rightBlock)) {}

    // copy-paste from above, keep in sync
    using base = Block<MergeByIndex<Left, OutId, Right, InId>>;

    mutable Left  _leftBlock;
    mutable Right _rightBlock;

    // merged_work_chunk_size, that's what friends are for
    friend base;

    template<BlockLike, std::size_t, BlockLike, std::size_t>
    friend class MergeByIndex;

private:
    static constexpr std::size_t merged_work_chunk_size() noexcept { return std::min({traits::block::stream_input_ports<Right>::template apply<traits::port::max_samples>::value, traits::block::stream_output_ports<Left>::template apply<traits::port::max_samples>::value, detail::subblockChunkSize<Left>(), detail::subblockChunkSize<Right>()}); }

    template<std::size_t I>
    constexpr auto apply_left(auto&& input_tuple) const noexcept {
        return [&]<std::size_t... Is>(std::index_sequence<Is...>) { return _leftBlock.processOne(std::get<Is>(std::forward<decltype(input_tuple)>(input_tuple))...); }(std::make_index_sequence<I>());
    }

    template<std::size_t I, std::size_t J>
    constexpr auto apply_right(auto&& input_tuple, auto&& tmp) const noexcept {
        return [&]<std::size_t... Is, std::size_t... Js>(std::index_sequence<Is...>, std::index_sequence<Js...>) {
            constexpr std::size_t first_offset  = traits::block::stream_input_port_types<Left>::size;
            constexpr std::size_t second_offset = traits::block::stream_input_port_types<Left>::size + sizeof...(Is);
            static_assert(second_offset + sizeof...(Js) == std::tuple_size_v<std::remove_cvref_t<decltype(input_tuple)>>);
            return _rightBlock.processOne(std::get<first_offset + Is>(std::forward<decltype(input_tuple)>(input_tuple))..., std::forward<decltype(tmp)>(tmp), std::get<second_offset + Js>(input_tuple)...);
        }(std::make_index_sequence<I>(), std::make_index_sequence<J>());
    }

public:
    constexpr MergeByIndex(Left&& l, Right&& r) : _leftBlock(std::move(l)), _rightBlock(std::move(r)) {}
    explicit constexpr MergeByIndex(gr::property_map init = {}) {
        detail::forwardSettings(_leftBlock, init);
        detail::forwardSettings(_rightBlock, init);
        detail::forwardNestedSettings(_leftBlock, init, "leftBlock");
        detail::forwardNestedSettings(_rightBlock, init, "rightBlock");
    }

    void stateChanged(lifecycle::State newState) {
        if (auto result = _leftBlock.changeStateTo(newState); !result) {
            this->emitErrorMessage("MergeByIndex::stateChanged(leftBlock)", result.error());
        }
        if (auto result = _rightBlock.changeStateTo(newState); !result) {
            this->emitErrorMessage("MergeByIndex::stateChanged(rightBlock)", result.error());
        }
    }

    void settingsChanged(const gr::property_map& /*oldSettings*/, const gr::property_map& newSettings) {
        detail::forwardSettings(_leftBlock, newSettings);
        detail::forwardSettings(_rightBlock, newSettings);
    }

    template<meta::any_simd... Ts>
    requires traits::block::can_processOne_simd<Left> and traits::block::can_processOne_simd<Right>
    constexpr meta::simdize<ReturnType, (0, ..., Ts::size())> processOne(const Ts&... inputs) const {
        static_assert(traits::block::stream_output_port_types<Left>::size == 1, "TODO: SIMD for multiple output ports not implemented yet");
        return apply_right<InId, traits::block::stream_input_port_types<Right>::size() - InId - 1>(std::tie(inputs...), apply_left<traits::block::stream_input_port_types<Left>::size()>(std::tie(inputs...)));
    }

    constexpr auto processOne(meta::constexpr_value auto N) const
    requires traits::block::can_processOne_simd<Right> and (traits::block::stream_input_port_types<Left>::size() == 0 or
                                                               requires(Left& l) {
                                                                   { l.processOne(N) };
                                                               })
    {
        if constexpr (requires(Left& l) {
                          { l.processOne(N) };
                      }) {
            return _rightBlock.processOne(_leftBlock.processOne(N));
        } else {
            using LeftResult = typename traits::block::stream_return_type<Left>;
            using V          = meta::simdize<LeftResult, N>;
            alignas(stdx::memory_alignment_v<V>) LeftResult tmp[V::size()];
            for (std::size_t i = 0UZ; i < V::size(); ++i) {
                tmp[i] = _leftBlock.processOne();
            }
            return _rightBlock.processOne(V(tmp, stdx::vector_aligned));
        }
    }

    template<typename... Ts>
    // Nicer error messages for the following would be good, but not at the expense of breaking can_processOne_simd.
    requires(InputPortTypes::template are_equal<std::remove_cvref_t<Ts>...>)
    constexpr ReturnType processOne(Ts&&... inputs) const {
        // if (sizeof...(Ts) == 0) we could call `return processOne(integral_constant<size_t, width>)`. But if
        // the caller expects to process *one* sample (no inputs for the caller to explicitly
        // request simd), and we process more, we risk inconsistencies.
        if constexpr (traits::block::stream_output_port_types<Left>::size == 1) {
            // only the result from the right block needs to be returned
            return apply_right<InId, traits::block::stream_input_port_types<Right>::size() - InId - 1>(std::forward_as_tuple(std::forward<Ts>(inputs)...), apply_left<traits::block::stream_input_port_types<Left>::size()>(std::forward_as_tuple(std::forward<Ts>(inputs)...)));

        } else {
            // left produces a tuple
            auto left_out  = apply_left<traits::block::stream_input_port_types<Left>::size()>(std::forward_as_tuple(std::forward<Ts>(inputs)...));
            auto right_out = apply_right<InId, traits::block::stream_input_port_types<Right>::size() - InId - 1>(std::forward_as_tuple(std::forward<Ts>(inputs)...), std::move(std::get<OutId>(left_out)));

            if constexpr (traits::block::stream_output_port_types<Left>::size == 2 && traits::block::stream_output_port_types<Right>::size == 1) {
                return std::make_tuple(std::move(std::get<OutId ^ 1>(left_out)), std::move(right_out));

            } else if constexpr (traits::block::stream_output_port_types<Left>::size == 2) {
                return std::tuple_cat(std::make_tuple(std::move(std::get<OutId ^ 1>(left_out))), std::move(right_out));

            } else if constexpr (traits::block::stream_output_port_types<Right>::size == 1) {
                return [&]<std::size_t... Is, std::size_t... Js>(std::index_sequence<Is...>, std::index_sequence<Js...>) { return std::make_tuple(std::move(std::get<Is>(left_out))..., std::move(std::get<OutId + 1 + Js>(left_out))..., std::move(right_out)); }(std::make_index_sequence<OutId>(), std::make_index_sequence<traits::block::stream_output_port_types<Left>::size - OutId - 1>());

            } else {
                return [&]<std::size_t... Is, std::size_t... Js, std::size_t... Ks>(std::index_sequence<Is...>, std::index_sequence<Js...>, std::index_sequence<Ks...>) { return std::make_tuple(std::move(std::get<Is>(left_out))..., std::move(std::get<OutId + 1 + Js>(left_out))..., std::move(std::get<Ks>(right_out)...)); }(std::make_index_sequence<OutId>(), std::make_index_sequence<traits::block::stream_output_port_types<Left>::size - OutId - 1>(), std::make_index_sequence<Right::output_port_types::size>());
            }
        }
    } // end:: processOne

    detail::GraphWithPortMaps graphWithPortMaps() const {
        Graph g;
        auto  leftPorts  = detail::addSubBlock(g, _leftBlock);
        auto  rightPorts = detail::addSubBlock(g, _rightBlock);

        auto& src   = leftPorts.outputPorts[OutId];
        auto& dst   = rightPorts.inputPorts[InId];
        std::ignore = g.addEdge(Edge(src.block, PortDefinition(src.portIndex), dst.block, PortDefinition(dst.portIndex), 0, 0, "merged"));

        // external outputs = Left outputs minus [OutId] + Right outputs
        std::vector<detail::PortRef> outputPorts;
        outputPorts.reserve(leftPorts.outputPorts.size() - 1 + rightPorts.outputPorts.size());
        for (std::size_t i = 0; i < leftPorts.outputPorts.size(); ++i) {
            if (i != OutId) {
                outputPorts.push_back(leftPorts.outputPorts[i]);
            }
        }
        for (auto& p : rightPorts.outputPorts) {
            outputPorts.push_back(p);
        }

        // external inputs = Left inputs + Right inputs minus [InId]
        std::vector<detail::PortRef> inputPorts = leftPorts.inputPorts;
        for (std::size_t i = 0; i < rightPorts.inputPorts.size(); ++i) {
            if (i != InId) {
                inputPorts.push_back(rightPorts.inputPorts[i]);
            }
        }

        return {std::move(g), std::move(outputPorts), std::move(inputPorts)};
    }

    Graph graph() const { return std::move(graphWithPortMaps().graph); }
};

namespace detail {
template<meta::fixed_string PortName, typename PortsTypeList>
consteval std::size_t checkedIndexForName() {
    constexpr std::size_t Id = meta::indexForName<PortName, PortsTypeList>();
    static_assert(Id != -1UZ);
    return Id;
}

} // namespace detail

/// Name-based alias for MergeByIndex — resolves port names to indices at compile time.
template<BlockLike Left, meta::fixed_string OutName, BlockLike Right, meta::fixed_string InName>
using Merge = MergeByIndex<Left, detail::checkedIndexForName<OutName, typename traits::block::stream_output_ports<Left>>(), //
    Right, detail::checkedIndexForName<InName, typename traits::block::stream_input_ports<Right>>()>;

/**
 * @brief Compile-time output-sign list for SplitMergeCombine paths.
 *
 * Usage: SplitMergeCombine<OutputSigns<+1.0f, -1.0f>, PathA, PathB>
 * Signs beyond the number of paths are ignored; missing signs default to +1.
 */
template<auto... Vs>
struct OutputSigns {};

namespace detail {

template<typename T>
struct SplitMergeDisplayBlock : gr::Block<SplitMergeDisplayBlock<T>> {
    gr::PortIn<T>  in;
    gr::PortOut<T> out;
    GR_MAKE_REFLECTABLE(SplitMergeDisplayBlock, in, out);
    explicit SplitMergeDisplayBlock(gr::property_map init = {}) : gr::Block<SplitMergeDisplayBlock<T>>(std::move(init)) {}
    [[nodiscard]] constexpr T processOne(T x) const noexcept { return x; }
};

template<typename T>
std::shared_ptr<BlockModel> makeSplitMergeDisplayNode(const std::string& label) {
    auto* wrapper            = new BlockWrapper<SplitMergeDisplayBlock<T>>();
    wrapper->blockRef().name = label;
    return std::shared_ptr<BlockModel>{static_cast<BlockModel*>(wrapper)};
}

template<BlockLike... Paths>
struct SplitMergeCombineTraits {
    static_assert(sizeof...(Paths) >= 2, "at least 2 paths required");

    using FirstPath  = std::tuple_element_t<0, std::tuple<Paths...>>;
    using InputType  = typename traits::block::stream_input_port_types<FirstPath>::template at<0>;
    using OutputType = typename traits::block::stream_output_port_types<FirstPath>::template at<0>;

    static_assert(((traits::block::stream_input_port_types<Paths>::size == 1) && ...), "each path must have exactly 1 stream input");
    static_assert(((traits::block::stream_output_port_types<Paths>::size == 1) && ...), "each path must have exactly 1 stream output");
    static_assert(((std::is_same_v<InputType, typename traits::block::stream_input_port_types<Paths>::template at<0>>) && ...), "all paths must have the same input type");
    static_assert(((std::is_same_v<OutputType, typename traits::block::stream_output_port_types<Paths>::template at<0>>) && ...), "all paths must have the same output type");

    static constexpr std::size_t mergedWorkChunkSize() noexcept { return std::min({subblockChunkSize<Paths>()..., traits::block::stream_input_ports<Paths>::template apply<traits::port::max_samples>::value..., traits::block::stream_output_ports<Paths>::template apply<traits::port::max_samples>::value...}); }
};
} // namespace detail

/**
 * @brief Compile-time fan-out merge: input is copied to N parallel paths, outputs are summed.
 *
 *               ┌──────────┐
 *         ┌─in─>│  Path0   ├─out──┐
 *         │     └──────────┘      │
 * ─ in ──>┤     ┌──────────┐      ├──(+)──> out
 *         │ in─>│  Path1   ├─out──┘
 *         └─    └──────────┘
 *         ...   (variadic)
 *
 * Each path must be a 1-in/1-out BlockLike. All inputs must share the same InputType and all
 * outputs the same OutputType (which may differ for converting paths). The splitter (identity
 * copy) and combiner (addition) are implicit — no separate blocks needed.
 *
 * Optional OutputSigns prescribe per-path sign/weight before summation:
 *   SplitMergeCombine<PathA, PathB>                         — all signs = +1
 *   SplitMergeCombine<OutputSigns<+1.0f, -1.0f>, PathA, PathB>  — PathA: +1, PathB: -1
 */
template<typename... Args>
struct SplitMergeCombine;

// specialisation: no OutputSigns — all signs default to +1 (keep in sync with OutputSigns specialisation below)
template<BlockLike... Paths>
struct SplitMergeCombine<Paths...> : Block<SplitMergeCombine<Paths...>> {
    using Traits     = detail::SplitMergeCombineTraits<Paths...>;
    using InputType  = typename Traits::InputType;
    using OutputType = typename Traits::OutputType;

    template<std::size_t>
    static constexpr OutputType signOf() {
        return OutputType(1);
    }

    gr::PortIn<InputType>   in;
    gr::PortOut<OutputType> out;

    GR_MAKE_REFLECTABLE(SplitMergeCombine, in, out);

    mutable std::tuple<Paths...> _paths;

    gr::meta::immutable<std::string> unique_name = std::format("SplitMergeCombine<{}>#{}", gr::meta::type_name<typename Traits::FirstPath>(), this->unique_id);

    SplitMergeCombine(const SplitMergeCombine&)            = delete;
    SplitMergeCombine& operator=(const SplitMergeCombine&) = delete;
    SplitMergeCombine& operator=(SplitMergeCombine&&)      = delete;

    SplitMergeCombine(SplitMergeCombine&& other) noexcept(std::is_nothrow_move_constructible_v<std::tuple<Paths...>>) : _paths(std::move(other._paths)) {}

    explicit constexpr SplitMergeCombine(gr::property_map init = {}) {
        [&]<std::size_t... Is>(std::index_sequence<Is...>) {
            (detail::forwardSettings(std::get<Is>(_paths), init), ...);
            (detail::forwardNestedSettings(std::get<Is>(_paths), init, std::format("path{}", Is)), ...);
        }(std::make_index_sequence<sizeof...(Paths)>());
    }

    template<std::size_t I>
    constexpr auto& path() const {
        return std::get<I>(_paths);
    }

    template<std::size_t I>
    constexpr auto& path() {
        return std::get<I>(_paths);
    }

    void stateChanged(lifecycle::State newState) {
        [&]<std::size_t... Is>(std::index_sequence<Is...>) {
            (
                [&] {
                    if (auto result = std::get<Is>(_paths).changeStateTo(newState); !result) {
                        this->emitErrorMessage(std::format("SplitMergeCombine::stateChanged(path{})", Is), result.error());
                    }
                }(),
                ...);
        }(std::make_index_sequence<sizeof...(Paths)>());
    }

    void settingsChanged(const gr::property_map& /*oldSettings*/, const gr::property_map& newSettings) {
        [&]<std::size_t... Is>(std::index_sequence<Is...>) { (detail::forwardSettings(std::get<Is>(_paths), newSettings), ...); }(std::make_index_sequence<sizeof...(Paths)>());
    }

    [[nodiscard]] constexpr OutputType processOne(InputType x) const noexcept {
        return [&]<std::size_t... Is>(std::index_sequence<Is...>) { return ((signOf<Is>() * std::get<Is>(_paths).processOne(x)) + ...); }(std::make_index_sequence<sizeof...(Paths)>());
    }

    detail::GraphWithPortMaps graphWithPortMaps() const {
        Graph g;
        auto  fanOut = detail::makeSplitMergeDisplayNode<InputType>("FanOut");
        auto  sum    = detail::makeSplitMergeDisplayNode<OutputType>("Sum");
        g.addBlock(fanOut, false);
        g.addBlock(sum, false);
        [&]<std::size_t... Is>(std::index_sequence<Is...>) {
            (([&] {
                auto  ports = detail::addSubBlock(g, std::get<Is>(_paths));
                auto& pIn   = ports.inputPorts[0];
                auto& pOut  = ports.outputPorts[0];
                std::ignore = g.addEdge(Edge(fanOut, PortDefinition(std::size_t(0)), pIn.block, PortDefinition(pIn.portIndex), 0, 0, "split"));
                std::ignore = g.addEdge(Edge(pOut.block, PortDefinition(pOut.portIndex), sum, PortDefinition(std::size_t(0)), 0, 0, "sum"));
            }()),
                ...);
        }(std::make_index_sequence<sizeof...(Paths)>());
        return {std::move(g), {{sum, 0}}, {{fanOut, 0}}};
    }

    Graph graph() const { return std::move(graphWithPortMaps().graph); }

private:
    using base = Block<SplitMergeCombine<Paths...>>;
    friend base;

    static constexpr std::size_t merged_work_chunk_size() noexcept { return Traits::mergedWorkChunkSize(); }
};

// specialisation: with OutputSigns — per-path sign/weight before summation (keep in sync with no-signs specialisation above)
template<auto... Vs, BlockLike... Paths>
struct SplitMergeCombine<OutputSigns<Vs...>, Paths...> : Block<SplitMergeCombine<OutputSigns<Vs...>, Paths...>> {
    using Traits     = detail::SplitMergeCombineTraits<Paths...>;
    using InputType  = typename Traits::InputType;
    using OutputType = typename Traits::OutputType;

    static constexpr std::array<OutputType, sizeof...(Vs)> kSigns = {static_cast<OutputType>(Vs)...};

    template<std::size_t I>
    static constexpr OutputType signOf() {
        if constexpr (I < kSigns.size()) {
            return kSigns[I];
        } else {
            return OutputType(1);
        }
    }

    gr::PortIn<InputType>   in;
    gr::PortOut<OutputType> out;

    GR_MAKE_REFLECTABLE(SplitMergeCombine, in, out);

    mutable std::tuple<Paths...> _paths;

    gr::meta::immutable<std::string> unique_name = std::format("SplitMergeCombine<OutputSigns,{}>#{}", gr::meta::type_name<typename Traits::FirstPath>(), this->unique_id);

    SplitMergeCombine(const SplitMergeCombine&)            = delete;
    SplitMergeCombine& operator=(const SplitMergeCombine&) = delete;
    SplitMergeCombine& operator=(SplitMergeCombine&&)      = delete;

    SplitMergeCombine(SplitMergeCombine&& other) noexcept(std::is_nothrow_move_constructible_v<std::tuple<Paths...>>) : _paths(std::move(other._paths)) {}

    explicit constexpr SplitMergeCombine(gr::property_map init = {}) {
        [&]<std::size_t... Is>(std::index_sequence<Is...>) {
            (detail::forwardSettings(std::get<Is>(_paths), init), ...);
            (detail::forwardNestedSettings(std::get<Is>(_paths), init, std::format("path{}", Is)), ...);
        }(std::make_index_sequence<sizeof...(Paths)>());
    }

    template<std::size_t I>
    constexpr auto& path() const {
        return std::get<I>(_paths);
    }

    template<std::size_t I>
    constexpr auto& path() {
        return std::get<I>(_paths);
    }

    void stateChanged(lifecycle::State newState) {
        [&]<std::size_t... Is>(std::index_sequence<Is...>) {
            (
                [&] {
                    if (auto result = std::get<Is>(_paths).changeStateTo(newState); !result) {
                        this->emitErrorMessage(std::format("SplitMergeCombine::stateChanged(path{})", Is), result.error());
                    }
                }(),
                ...);
        }(std::make_index_sequence<sizeof...(Paths)>());
    }

    void settingsChanged(const gr::property_map& /*oldSettings*/, const gr::property_map& newSettings) {
        [&]<std::size_t... Is>(std::index_sequence<Is...>) { (detail::forwardSettings(std::get<Is>(_paths), newSettings), ...); }(std::make_index_sequence<sizeof...(Paths)>());
    }

    [[nodiscard]] constexpr OutputType processOne(InputType x) const noexcept {
        return [&]<std::size_t... Is>(std::index_sequence<Is...>) { return ((signOf<Is>() * std::get<Is>(_paths).processOne(x)) + ...); }(std::make_index_sequence<sizeof...(Paths)>());
    }

    detail::GraphWithPortMaps graphWithPortMaps() const {
        Graph g;
        auto  fanOut = detail::makeSplitMergeDisplayNode<InputType>("FanOut");
        auto  sum    = detail::makeSplitMergeDisplayNode<OutputType>("Sum");
        g.addBlock(fanOut, false);
        g.addBlock(sum, false);
        [&]<std::size_t... Is>(std::index_sequence<Is...>) {
            (([&] {
                auto  ports = detail::addSubBlock(g, std::get<Is>(_paths));
                auto& pIn   = ports.inputPorts[0];
                auto& pOut  = ports.outputPorts[0];
                std::ignore = g.addEdge(Edge(fanOut, PortDefinition(std::size_t(0)), pIn.block, PortDefinition(pIn.portIndex), 0, 0, "split"));
                std::ignore = g.addEdge(Edge(pOut.block, PortDefinition(pOut.portIndex), sum, PortDefinition(std::size_t(0)), 0, 0, "sum"));
            }()),
                ...);
        }(std::make_index_sequence<sizeof...(Paths)>());
        return {std::move(g), {{sum, 0}}, {{fanOut, 0}}};
    }

    Graph graph() const { return std::move(graphWithPortMaps().graph); }

private:
    using base = Block<SplitMergeCombine<OutputSigns<Vs...>, Paths...>>;
    friend base;

    static constexpr std::size_t merged_work_chunk_size() noexcept { return Traits::mergedWorkChunkSize(); }
};

template<BlockLike Forward, std::size_t ForwardOutputPortIndex, //
    BlockLike Feedback, std::size_t FeedbackOutputPortIndex,    //
    std::size_t ForwardFeedbackInputPortIndex,                  //
    typename Monitor>
class FeedbackMergeBase {
public:
    static_assert(traits::block::stream_input_port_types<Feedback>::size == 1, "Feedback block needs to have only one input port");
    static_assert(traits::block::stream_input_port_types<Forward>::size >= 2, "Forward block must have at least 2 input ports");

    using MergeConnectionForwardOutputType     = typename traits::block::stream_output_port_types<Forward>::template at<ForwardOutputPortIndex>;
    using MergeConnectionFeedbackInputType     = typename traits::block::stream_input_port_types<Feedback>::template at<0>;
    using FeedbackConnectionFeedbackOutputType = typename traits::block::stream_output_port_types<Feedback>::template at<FeedbackOutputPortIndex>;
    using FeedbackConnectionForwardInputType   = typename traits::block::stream_input_port_types<Forward>::template at<ForwardFeedbackInputPortIndex>;
    using MergeConnectionPortType              = MergeConnectionForwardOutputType;
    using FeedbackConnectionPortType           = FeedbackConnectionFeedbackOutputType;

    static_assert(std::is_same_v<MergeConnectionForwardOutputType, MergeConnectionFeedbackInputType>, "The chosen output port of Forward block needs to have the same type as the Feedback input port type");
    static_assert(std::is_same_v<FeedbackConnectionFeedbackOutputType, FeedbackConnectionForwardInputType>, "The chosen output port of Feedback block needs to have the same type as the chosen Forward input port type");

    using SelfInputPortDescriptors = meta::remove_at<ForwardFeedbackInputPortIndex, typename traits::block::stream_input_ports<Forward>>;
    using OverridePortList         = meta::concat<                                              //
        meta::transform_types<detail::to_forward_descriptor, SelfInputPortDescriptors>, //
        meta::transform_types<detail::to_forward_descriptor, typename traits::block::stream_output_ports<Forward>>>;
    using SelfInputPortTypes       = meta::remove_at<ForwardFeedbackInputPortIndex, typename traits::block::stream_input_port_types<Forward>>;
    using SelfOutputPortTypes      = typename traits::block::stream_output_port_types<Forward>;
    using ReturnType               = typename SelfOutputPortTypes::tuple_or_type;

    mutable Forward                    _forward;
    mutable Feedback                   _feedback;
    mutable FeedbackConnectionPortType _state{};

    [[no_unique_address]] mutable std::conditional_t<std::is_same_v<void, Monitor>, detail::Empty, Monitor> _monitor;

    FeedbackMergeBase(const FeedbackMergeBase&)            = delete;
    FeedbackMergeBase& operator=(const FeedbackMergeBase&) = delete;
    FeedbackMergeBase& operator=(FeedbackMergeBase&&)      = delete;

    FeedbackMergeBase(FeedbackMergeBase&& other) noexcept : _forward(std::move(other._forward)), _feedback(std::move(other._feedback)), _state(std::move(other._state)) {}

    constexpr FeedbackMergeBase(Forward&& fwd, Feedback&& fbk) : _forward(std::move(fwd)), _feedback(std::move(fbk)) {}

    explicit FeedbackMergeBase(gr::property_map init = {}) {
        detail::forwardSettings(_forward, init);
        detail::forwardSettings(_feedback, init);
        detail::forwardNestedSettings(_forward, init, "forward");
        detail::forwardNestedSettings(_feedback, init, "feedback");
        if constexpr (!std::is_same_v<void, Monitor>) {
            detail::forwardNestedSettings(_monitor, init, "monitor");
        }
    }

    void settingsChanged(const gr::property_map& /*oldSettings*/, const gr::property_map& newSettings) {
        detail::forwardSettings(_forward, newSettings);
        detail::forwardSettings(_feedback, newSettings);
    }

public:
    template<typename... Ts>
    requires(SelfInputPortTypes::template are_equal<std::remove_cvref_t<Ts>...>)
    constexpr ReturnType processOne(Ts&&... inputs) const noexcept {
        auto forwardInputTuple = std::forward_as_tuple(std::forward<Ts>(inputs)...);

        auto output = [&]<std::size_t... BeforeIdx, std::size_t... AfterIdx>(std::index_sequence<BeforeIdx...>, std::index_sequence<AfterIdx...>) {
            constexpr std::size_t afterOffset = ForwardFeedbackInputPortIndex + 1;
            return _forward.processOne(std::get<BeforeIdx>(forwardInputTuple)..., _state, std::get<afterOffset + AfterIdx - 1>(forwardInputTuple)...);
        }(std::make_index_sequence<ForwardFeedbackInputPortIndex>(), std::make_index_sequence<sizeof...(Ts) - ForwardFeedbackInputPortIndex>());

        if constexpr (std::is_same_v<void, Monitor>) {
            _state = _feedback.processOne(output);
        } else {
            _state = _feedback.processOne(_monitor.processOne(output));
        }
        return output;
    }

    detail::GraphWithPortMaps graphWithPortMaps() const {
        Graph g;
        auto  fwdPorts = detail::addSubBlock(g, _forward);
        auto  fbkPorts = detail::addSubBlock(g, _feedback);

        auto& fwdOut = fwdPorts.outputPorts[ForwardOutputPortIndex];
        auto& fbkIn  = fbkPorts.inputPorts[0];
        auto& fbkOut = fbkPorts.outputPorts[FeedbackOutputPortIndex];
        auto& fwdIn  = fwdPorts.inputPorts[ForwardFeedbackInputPortIndex];

        if constexpr (!std::is_same_v<void, Monitor>) {
            auto  monPorts = detail::addSubBlock(g, _monitor);
            auto& monIn    = monPorts.inputPorts[0];
            auto& monOut   = monPorts.outputPorts[0];
            std::ignore    = g.addEdge(Edge(fwdOut.block, PortDefinition(fwdOut.portIndex), monIn.block, PortDefinition(monIn.portIndex), 0, 0, "to_monitor"));
            std::ignore    = g.addEdge(Edge(monOut.block, PortDefinition(monOut.portIndex), fbkIn.block, PortDefinition(fbkIn.portIndex), 0, 0, "monitor_to_feedback"));
        } else {
            std::ignore = g.addEdge(Edge(fwdOut.block, PortDefinition(fwdOut.portIndex), fbkIn.block, PortDefinition(fbkIn.portIndex), 0, 0, "forward_to_feedback"));
        }
        std::ignore = g.addEdge(Edge(fbkOut.block, PortDefinition(fbkOut.portIndex), fwdIn.block, PortDefinition(fwdIn.portIndex), 0, 0, "feedback_loop"));

        // external outputs = Forward's outputs
        std::vector<detail::PortRef> outputPorts = fwdPorts.outputPorts;

        // external inputs = Forward's inputs minus [ForwardFeedbackInputPortIndex]
        std::vector<detail::PortRef> inputPorts;
        inputPorts.reserve(fwdPorts.inputPorts.size() - 1);
        for (std::size_t i = 0; i < fwdPorts.inputPorts.size(); ++i) {
            if (i != ForwardFeedbackInputPortIndex) {
                inputPorts.push_back(fwdPorts.inputPorts[i]);
            }
        }

        return {std::move(g), std::move(outputPorts), std::move(inputPorts)};
    }

    Graph graph() const { return std::move(graphWithPortMaps().graph); }
};

/**
 * Feedback merge for blocks that feed data previously generated
 * to one of the ports.
 *
 *           Forward
 *           adder       ┌─────────────────────> out of FeedbackMerge
 *           ┌────┐      │
 * ─────in1─>┤    │      │      Feedback
 *           │    ├─out─>┤      scale
 *    ┌─in2─>┤    │      │      ┌────┐
 *    │      └────┘      └──in─>┤    ├─out──┐
 *    │                         └────┘      │
 *    │                                     │
 *    │             ┌──────────┐            │
 *    └─────────────┤          │<───────────┘
 *                  └──────────┘
 *                  Monitor (optional)
 */
template<BlockLike Forward, std::size_t ForwardOutputPortIndex, //
    BlockLike Feedback, std::size_t FeedbackOutputPortIndex,    //
    std::size_t ForwardFeedbackInputPortIndex,                  //
    typename Monitor>
class FeedbackMergeByIndex : public Block<FeedbackMergeByIndex<Forward, ForwardOutputPortIndex,    //
                                 Feedback, FeedbackOutputPortIndex, ForwardFeedbackInputPortIndex, //
                                 Monitor>>,                                                        //
                             public FeedbackMergeBase<Forward, ForwardOutputPortIndex,             //
                                 Feedback, FeedbackOutputPortIndex, ForwardFeedbackInputPortIndex, //
                                 Monitor> {
    using impl_t = FeedbackMergeBase<Forward, ForwardOutputPortIndex,     //
        Feedback, FeedbackOutputPortIndex, ForwardFeedbackInputPortIndex, //
        Monitor>;

public:
    gr::meta::immutable<std::string> unique_name = std::format("FeedbackMergeByIndex<{}:{},{}:{},feedback_to:{}>#{}", gr::meta::type_name<Forward>(), ForwardOutputPortIndex, gr::meta::type_name<Feedback>(), FeedbackOutputPortIndex, ForwardFeedbackInputPortIndex, this->unique_id);

    using OverridePortList = typename impl_t::OverridePortList;
    using impl_t::impl_t;
    using impl_t::processOne;
    using impl_t::settingsChanged;

    void stateChanged(lifecycle::State newState) {
        if (auto result = this->_forward.changeStateTo(newState); !result) {
            this->emitErrorMessage("FeedbackMerge::stateChanged(forward)", result.error());
        }
        if (auto result = this->_feedback.changeStateTo(newState); !result) {
            this->emitErrorMessage("FeedbackMerge::stateChanged(feedback)", result.error());
        }
    }

    GR_MAKE_REFLECTABLE(FeedbackMergeByIndex);
};

template<BlockLike Forward, meta::fixed_string ForwardOutputPortName, //
    BlockLike Feedback, meta::fixed_string FeedbackOutputPortName,    //
    meta::fixed_string ForwardFeedbackInputPortName, typename Monitor = void>
using FeedbackMerge = FeedbackMergeByIndex<                                                                                 //
    Forward, detail::checkedIndexForName<ForwardOutputPortName, typename traits::block::stream_output_ports<Forward>>(),    //
    Feedback, detail::checkedIndexForName<FeedbackOutputPortName, typename traits::block::stream_output_ports<Feedback>>(), //
    detail::checkedIndexForName<ForwardFeedbackInputPortName, typename traits::block::stream_input_ports<Forward>>(), Monitor>;

/**
 * Feedback merge for blocks that feed data previously generated
 * to one of the ports, with splitOut port to allow connecting other
 * blocks to the output of the feedback block.
 *
 * FeedbackMergeWithTap<Adder, "out", Scale<0.2f>, "out", "in2">;
 *
 *           Forward
 *           adder       ┌──────────────────────> out of FeedbackMergeWithTap
 *           ┌────┐      │
 * ─────in1─>┤    │      │      Feedback
 *           │    ├─out─>┤      scale       ┌--─> splitOut
 *    ┌─in2─>┤    │      │      ┌────┐      │
 *    │      └────┘      └──in─>┤    ├─out─>┤
 *    │                         └────┘      │
 *    └──────────────────<──────────────────┘
 *
 */
template<BlockLike Forward, std::size_t ForwardOutputPortIndex, //
    BlockLike Feedback, std::size_t FeedbackOutputPortIndex,    //
    std::size_t ForwardFeedbackInputPortIndex,                  //
    typename Monitor>
class FeedbackMergeWithTapByIndex : public Block<FeedbackMergeWithTapByIndex<Forward, ForwardOutputPortIndex, Feedback, FeedbackOutputPortIndex, ForwardFeedbackInputPortIndex, Monitor>>, //
                                    public FeedbackMergeBase<Forward, ForwardOutputPortIndex, Feedback, FeedbackOutputPortIndex, ForwardFeedbackInputPortIndex, Monitor> {
    using impl_t = FeedbackMergeBase<Forward, ForwardOutputPortIndex, Feedback, FeedbackOutputPortIndex, ForwardFeedbackInputPortIndex, Monitor>;
    using this_t = FeedbackMergeWithTapByIndex<Forward, ForwardOutputPortIndex, Feedback, FeedbackOutputPortIndex, ForwardFeedbackInputPortIndex, Monitor>;

public:
    gr::PortOut<typename impl_t::FeedbackConnectionPortType> splitOut;

    GR_MAKE_REFLECTABLE(FeedbackMergeWithTapByIndex, splitOut);

    gr::meta::immutable<std::string> unique_name = std::format("FeedbackMergeWithTapByIndex<{}:{},{}:{},feedback_to:{}>#{}", gr::meta::type_name<Forward>(), ForwardOutputPortIndex, gr::meta::type_name<Feedback>(), FeedbackOutputPortIndex, ForwardFeedbackInputPortIndex, this->unique_id);

    using OverridePortList = meta::concat<typename impl_t::OverridePortList,                    //
        gr::meta::typelist<                                                                     //
            gr::detail::PortDescriptor<typename impl_t::FeedbackConnectionPortType, "splitOut", //
                gr::PortType::STREAM, gr::PortDirection::OUTPUT, gr::detail::SinglePort,        //
                /*KindExtraData=*/0, /*MemberIdx=*/refl::data_member_count<Block<this_t>>>>>;

    using impl_t::impl_t;
    using impl_t::settingsChanged;

    void stateChanged(lifecycle::State newState) {
        if (auto result = this->_forward.changeStateTo(newState); !result) {
            this->emitErrorMessage("FeedbackMergeWithTap::stateChanged(forward)", result.error());
        }
        if (auto result = this->_feedback.changeStateTo(newState); !result) {
            this->emitErrorMessage("FeedbackMergeWithTap::stateChanged(feedback)", result.error());
        }
    }

    // needed by CRTP Block machinery for source-like detection
    constexpr auto processOne() { return typename impl_t::FeedbackConnectionPortType{}; }

    template<typename... Ts>
    requires(impl_t::SelfInputPortTypes::template are_equal<std::remove_cvref_t<Ts>...>)
    constexpr auto processOne(Ts&&... inputs) const noexcept {
        auto result = impl_t::processOne(std::forward<Ts>(inputs)...);

        if constexpr (meta::is_instantiation_of<typename impl_t::ReturnType, std::tuple>) {
            return std::tuple_cat(result, std::make_tuple(impl_t::_state));
        } else {
            return std::tuple(result, impl_t::_state);
        }
    }
};

template<BlockLike Forward, meta::fixed_string ForwardOutputPortName, //
    BlockLike Feedback, meta::fixed_string FeedbackOutputPortName,    //
    meta::fixed_string ForwardFeedbackInputPortName,
    typename Monitor = void>
using FeedbackMergeWithTap = FeedbackMergeWithTapByIndex<                                                                   //
    Forward, detail::checkedIndexForName<ForwardOutputPortName, typename traits::block::stream_output_ports<Forward>>(),    //
    Feedback, detail::checkedIndexForName<FeedbackOutputPortName, typename traits::block::stream_output_ports<Feedback>>(), //
    detail::checkedIndexForName<ForwardFeedbackInputPortName, typename traits::block::stream_input_ports<Forward>>(),       //
    Monitor>;

} // namespace gr

#endif // GNURADIO_BLOCK_MERGING_HPP
