#ifndef GNURADIO_BLOCK_MODEL_HPP
#define GNURADIO_BLOCK_MODEL_HPP

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/LifeCycle.hpp>
#include <gnuradio-4.0/Port.hpp>
#include <gnuradio-4.0/Settings.hpp>
#include <gnuradio-4.0/thread/thread_pool.hpp>

#include <charconv>

namespace gr {
class BlockModel;

struct PortDefinition {
    struct IndexBased {
        std::size_t topLevel;
        std::size_t subIndex;

        bool operator==(const IndexBased& other) const { return (topLevel == other.topLevel) && (subIndex == other.subIndex); }
    };

    struct StringBased {
        std::string name;

        bool operator==(const StringBased& other) const { return (name == other.name); }
    };

    std::variant<IndexBased, StringBased> definition;

    constexpr PortDefinition(std::size_t _topLevel, std::size_t _subIndex = meta::invalid_index) : definition(IndexBased{_topLevel, _subIndex}) {}
    constexpr PortDefinition(std::string name) : definition(StringBased(std::move(name))) {}
    bool operator==(const PortDefinition& other) const { return (definition == other.definition); }
};
} // namespace gr

namespace std { // needs to be defined in std namespace to be used e.g. in std::unordered_map
template<>
struct hash<gr::PortDefinition> {
    std::size_t operator()(const gr::PortDefinition& p) const noexcept {
        if (std::holds_alternative<gr::PortDefinition::IndexBased>(p.definition)) {
            const auto& def = std::get<gr::PortDefinition::IndexBased>(p.definition);
            std::size_t h1  = std::hash<std::size_t>{}(def.topLevel);
            std::size_t h2  = std::hash<std::size_t>{}(def.subIndex);
            return h1 ^ (h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2)); // hash_combine
        } else {                                                   // string-based
            const auto& def = std::get<gr::PortDefinition::StringBased>(p.definition);
            return std::hash<std::string>{}(def.name);
        }
    }
};
} // namespace std

namespace gr {

struct Edge {
    enum class EdgeState { WaitingToBeConnected, Connected, Overridden, ErrorConnecting, PortNotFound, IncompatiblePorts, Unknown };

    // Member variables that are controlled by the graph and scheduler
    std::shared_ptr<BlockModel> _sourceBlock;
    std::shared_ptr<BlockModel> _destinationBlock;
    PortDefinition              _sourcePortDefinition;
    PortDefinition              _destinationPortDefinition;
    EdgeState                   _state            = EdgeState::WaitingToBeConnected;
    std::size_t                 _actualBufferSize = -1UZ;
    PortType                    _edgeType         = PortType::ANY;
    DynamicPort*                _sourcePort       = nullptr; /// non-owning reference
    DynamicPort*                _destinationPort  = nullptr; /// non-owning reference

    // User-controlled member variables
    std::size_t  _minBufferSize;
    std::int32_t _weight = 0;
    std::string  _name   = "unnamed edge"; // custom edge name

    Edge() = delete;

    explicit Edge(std::shared_ptr<BlockModel> sourceBlock, PortDefinition sourcePortDefinition,               //
        std::shared_ptr<BlockModel> destinationBlock, PortDefinition destinationPortDefinition,               //
        std::size_t minBufferSize, std::int32_t weight, std::string name) noexcept                            //
        : _sourceBlock(sourceBlock), _destinationBlock(destinationBlock),                                     //
          _sourcePortDefinition(sourcePortDefinition), _destinationPortDefinition(destinationPortDefinition), //
          _minBufferSize(minBufferSize), _weight(weight), _name(std::move(name)) {}

    [[nodiscard]] constexpr const std::shared_ptr<BlockModel>& sourceBlock() const noexcept { return _sourceBlock; }
    [[nodiscard]] constexpr const std::shared_ptr<BlockModel>& destinationBlock() const noexcept { return _destinationBlock; }
    [[nodiscard]] PortDefinition                               sourcePortDefinition() const noexcept { return _sourcePortDefinition; }
    [[nodiscard]] PortDefinition                               destinationPortDefinition() const noexcept { return _destinationPortDefinition; }
    [[nodiscard]] constexpr EdgeState                          state() const noexcept { return _state; }

    [[nodiscard]] constexpr std::size_t      minBufferSize() const noexcept { return _minBufferSize; }
    [[nodiscard]] constexpr std::int32_t     weight() const noexcept { return _weight; }
    [[nodiscard]] constexpr std::string_view name() const noexcept { return _name; }

    constexpr void setMinBufferSize(std::size_t minBufferSize) { _minBufferSize = minBufferSize; }
    constexpr void setWeight(std::int32_t weight) { _weight = weight; }
    constexpr void setName(std::string name) { _name = std::move(name); }

    constexpr std::size_t bufferSize() const { return _actualBufferSize; }
    constexpr std::size_t nReaders() const { return _sourcePort ? _sourcePort->nReaders() : -1UZ; }
    constexpr std::size_t nWriters() const { return _destinationPort ? _destinationPort->nWriters() : -1UZ; }
    constexpr PortType    edgeType() const { return _edgeType; }

    constexpr bool hasSameSourcePort(const Edge& other) const noexcept { return sourceBlock() == other.sourceBlock() && sourcePortDefinition().definition == other.sourcePortDefinition().definition; }

    constexpr bool operator==(const Edge& other) const noexcept {
        return sourceBlock() == other.sourceBlock()                                                       //
               && destinationBlock() == other.destinationBlock()                                          //
               && sourcePortDefinition().definition == other.sourcePortDefinition().definition            //
               && destinationPortDefinition().definition == other.destinationPortDefinition().definition; //
    }
};

class BlockModel {
public:
    struct NamedPortCollection {
        std::string_view             name;
        std::vector<gr::DynamicPort> ports;

        [[nodiscard]] auto disconnect() {
            return std::ranges::count_if(ports, [](auto& port) { return port.disconnect() == ConnectionResult::FAILED; }) == 0 ? ConnectionResult::SUCCESS : ConnectionResult::FAILED;
        }
    };

    using DynamicPortOrCollection = std::variant<gr::DynamicPort, NamedPortCollection>;
    using DynamicPorts            = std::vector<DynamicPortOrCollection>;

protected:
    struct DynamicPortsLoader {
        using LoaderFn = void (*)(BlockModel*);

        LoaderFn    fn       = nullptr;
        BlockModel* instance = nullptr;

        void operator()() const {
            if (instance) {
                fn(instance);
            }
        }
    };

    bool               _dynamicPortsLoaded = false;
    DynamicPortsLoader _dynamicPortsLoader;
    DynamicPorts       _dynamicInputPorts;
    DynamicPorts       _dynamicOutputPorts;

    BlockModel() = default;

    [[nodiscard]] gr::DynamicPort& dynamicPortFromName(DynamicPorts& what, const std::string& name, std::source_location location = std::source_location::current()) {
        initDynamicPorts();

        if (auto separatorIt = std::ranges::find(name, '#'); separatorIt == name.end()) {
            auto it = std::ranges::find_if(what, [name](const DynamicPortOrCollection& portOrCollection) {
                const auto* port = std::get_if<gr::DynamicPort>(&portOrCollection);
                return port && port->name == name;
            });

            if (it == what.end()) {
                throw gr::exception(std::format("dynamicPortFromName([{}]) - Port {} not found in {}\n", what.size(), name, uniqueName()), location);
            }

            return std::get<gr::DynamicPort>(*it);
        } else {
            const std::string_view base(name.begin(), separatorIt);
            const std::string_view indexString(separatorIt + 1, name.end());
            std::size_t            index = -1UZ;
            auto [_, ec]                 = std::from_chars(indexString.data(), indexString.data() + indexString.size(), index);
            if (ec != std::errc()) {
                throw gr::exception(std::format("dynamicPortFromName([{}]) - Invalid index {} specified, needs to be an integer", what.size(), indexString), location);
            }

            auto collectionIt = std::ranges::find_if(what, [&base](const DynamicPortOrCollection& portOrCollection) {
                const auto* collection = std::get_if<NamedPortCollection>(&portOrCollection);
                return collection && collection->name == base;
            });

            if (collectionIt == what.cend()) {
                throw gr::exception(std::format("dynamicPortFromName([{}]) - Invalid name specified name={}, base={}\n", what.size(), name, base), location);
            }

            auto& collection = std::get<NamedPortCollection>(*collectionIt);

            if (index >= collection.ports.size()) {
                throw gr::exception(std::format("dynamicPortFromName([{}]) - Invalid index {} specified, out of range. Number of ports is {}", what.size(), index, collection.ports.size()), location);
            }

            return collection.ports[index];
        }
    }

    template<PortDirection direction, typename PortsVec>
    DynamicPort& dynamicPortByIndexImpl(PortsVec& portsVec, std::size_t topIndex, std::size_t subIndex, std::source_location loc) {
        using namespace std::string_view_literals;
        constexpr std::string_view which = (direction == PortDirection::INPUT ? "Input"sv : "Output"sv);
        initDynamicPorts();

        if (topIndex >= portsVec.size()) {
            throw gr::exception(std::format("dynamic{}Port(index: {}, subIndex: {}) - specified topIndex {} is out of range [0, {}]", which, topIndex, subIndex, topIndex, portsVec.size()), loc);
        }
        auto& entry = portsVec.at(topIndex);

        if (auto* collection = std::get_if<NamedPortCollection>(&entry)) {
            if (subIndex == meta::invalid_index) {
                throw gr::exception(std::format("invalid_argument: dynamic{}Port(index: {}, subIndex: {}) - Need to specify the index in the port collection for {}", which, topIndex, subIndex, collection->name), loc);
            }
            if (subIndex >= collection->ports.size()) {
                throw gr::exception(std::format("out_of_range: dynamic{}Port(index: {}, subIndex: {}) - sub-index out of range for {} (size={})", which, topIndex, subIndex, collection->name, collection->ports.size()), loc);
            }
            return collection->ports[subIndex];
        }

        auto* single = std::get_if<gr::DynamicPort>(&entry);
        if (!single) {
            throw gr::exception("variant construction failed", loc);
        }

        if (subIndex != meta::invalid_index) {
            throw gr::exception(std::format("invalid_argument: dynamic{}Port(index: {}, subIndex: {}) - specified sub-index for a normal port {}", which, topIndex, subIndex, single->name), loc);
        }
        return *single;
    }

public:
    BlockModel(const BlockModel&)             = delete;
    BlockModel& operator=(const BlockModel&)  = delete;
    BlockModel(BlockModel&& other)            = delete;
    BlockModel& operator=(BlockModel&& other) = delete;

    void initDynamicPorts() const {
        if (!_dynamicPortsLoaded) {
            _dynamicPortsLoader();
        }
    }

    MsgPortInBuiltin*  msgIn;
    MsgPortOutBuiltin* msgOut;

    static std::string portName(const DynamicPortOrCollection& portOrCollection) {
        return std::visit(meta::overloaded{                                          //
                              [](const gr::DynamicPort& port) { return port.name; }, //
                              [](const NamedPortCollection& namedCollection) { return std::string(namedCollection.name); }},
            portOrCollection);
    }

    [[nodiscard]] virtual std::span<std::shared_ptr<BlockModel>>       blocks() noexcept       = 0;
    [[nodiscard]] virtual std::span<const std::shared_ptr<BlockModel>> blocks() const noexcept = 0;
    [[nodiscard]] virtual std::span<Edge>                              edges() noexcept        = 0;
    [[nodiscard]] virtual std::span<const Edge>                        edges() const noexcept  = 0;

    DynamicPorts& dynamicInputPorts() {
        initDynamicPorts();
        return _dynamicInputPorts;
    }
    DynamicPorts& dynamicOutputPorts() {
        initDynamicPorts();
        return _dynamicOutputPorts;
    }

    [[nodiscard]] gr::DynamicPort& dynamicInputPort(const std::string& name, std::source_location location = std::source_location::current()) { return dynamicPortFromName(_dynamicInputPorts, name, location); }
    [[nodiscard]] gr::DynamicPort& dynamicOutputPort(const std::string& name, std::source_location location = std::source_location::current()) { return dynamicPortFromName(_dynamicOutputPorts, name, location); }
    [[nodiscard]] gr::DynamicPort& dynamicInputPort(std::size_t index, std::size_t subIndex = meta::invalid_index, std::source_location loc = std::source_location::current()) { return dynamicPortByIndexImpl<PortDirection::INPUT>(_dynamicInputPorts, index, subIndex, std::move(loc)); }
    [[nodiscard]] gr::DynamicPort& dynamicOutputPort(std::size_t index, std::size_t subIndex = meta::invalid_index, std::source_location loc = std::source_location::current()) { return dynamicPortByIndexImpl<PortDirection::OUTPUT>(_dynamicOutputPorts, index, subIndex, std::move(loc)); }

    [[nodiscard]] gr::DynamicPort& dynamicInputPort(PortDefinition definition, std::source_location location = std::source_location::current()) {
        return std::visit(meta::overloaded(                                                                                                                                                        //
                              [this, &location](const PortDefinition::IndexBased& _definition) -> DynamicPort& { return dynamicInputPort(_definition.topLevel, _definition.subIndex, location); }, //
                              [this, &location](const PortDefinition::StringBased& _definition) -> DynamicPort& { return dynamicInputPort(_definition.name, location); }),                         //
            definition.definition);
    }

    [[nodiscard]] gr::DynamicPort& dynamicOutputPort(PortDefinition definition, std::source_location location = std::source_location::current()) {
        return std::visit(meta::overloaded(                                                                                                                                                         //
                              [this, &location](const PortDefinition::IndexBased& _definition) -> DynamicPort& { return dynamicOutputPort(_definition.topLevel, _definition.subIndex, location); }, //
                              [this, &location](const PortDefinition::StringBased& _definition) -> DynamicPort& { return dynamicOutputPort(_definition.name, location); }),                         //
            definition.definition);
    }

    [[nodiscard]] std::size_t dynamicInputPortsSize(std::size_t parentIndex = meta::invalid_index) const {
        initDynamicPorts();
        if (parentIndex == meta::invalid_index) {
            return _dynamicInputPorts.size();
        } else {
            if (auto* portCollection = std::get_if<NamedPortCollection>(&_dynamicInputPorts.at(parentIndex))) {
                return portCollection->ports.size();
            } else {
                return meta::invalid_index;
            }
        }
    }

    [[nodiscard]] std::size_t dynamicOutputPortsSize(std::size_t parentIndex = meta::invalid_index) const {
        initDynamicPorts();
        if (parentIndex == meta::invalid_index) {
            return _dynamicOutputPorts.size();
        } else {
            if (auto* portCollection = std::get_if<NamedPortCollection>(&_dynamicOutputPorts.at(parentIndex))) {
                return portCollection->ports.size();
            } else {
                return meta::invalid_index;
            }
        }
    }

    std::size_t dynamicInputPortIndex(const std::string& name) const {
        initDynamicPorts();
        for (std::size_t i = 0; i < _dynamicInputPorts.size(); ++i) {
            if (auto* portCollection = std::get_if<NamedPortCollection>(&_dynamicInputPorts.at(i))) {
                if (portCollection->name == name) {
                    return i;
                }
            } else if (auto* port = std::get_if<gr::DynamicPort>(&_dynamicInputPorts.at(i))) {
                if (port->name == name) {
                    return i;
                }
            }
        }

        throw gr::exception(std::format("Port {} does not exist", name));
    }

    std::size_t dynamicOutputPortIndex(const std::string& name) const {
        initDynamicPorts();
        for (std::size_t i = 0; i < _dynamicOutputPorts.size(); ++i) {
            if (auto* portCollection = std::get_if<NamedPortCollection>(&_dynamicOutputPorts.at(i))) {
                if (portCollection->name == name) {
                    return i;
                }
            } else if (auto* port = std::get_if<gr::DynamicPort>(&_dynamicOutputPorts.at(i))) {
                if (port->name == name) {
                    return i;
                }
            }
        }

        throw gr::exception(std::format("Port {} does not exist", name));
    }

    virtual ~BlockModel() = default;

    /**
     * @brief to be called by scheduler->graph to initialise block
     */
    virtual void init(std::shared_ptr<gr::Sequence> progress, std::string_view ioThreadPool) = 0;

    /**
     * @brief returns scheduling hint that invoking the work(...) function may block on IO or system-calls
     */
    [[nodiscard]] virtual constexpr bool isBlocking() const noexcept = 0;

    /**
     * @brief change Block state (N.B. IDLE, INITIALISED, RUNNING, REQUESTED_STOP, REQUESTED_PAUSE, STOPPED, PAUSED, ERROR)
     * See enum description for details.
     */
    [[nodiscard]] virtual std::expected<void, Error> changeStateTo(lifecycle::State newState) noexcept = 0;

    /**
     * @brief Block state (N.B. IDLE, INITIALISED, RUNNING, REQUESTED_STOP, REQUESTED_PAUSE, STOPPED, PAUSED, ERROR)
     * See enum description for details.
     */
    [[nodiscard]] virtual lifecycle::State state() const noexcept = 0;

    /**
     * @brief user defined name
     */
    [[nodiscard]] virtual std::string_view name() const = 0;

    /**
     * @brief the type of the node as a string
     */
    [[nodiscard]] virtual std::string_view typeName() const = 0;

    /**
     * @brief user-defined name
     * N.B. may not be unique -> ::uniqueName
     */
    virtual void setName(std::string name) noexcept = 0;

    /**
     * @brief used to store static non-graph-processing information like Annotated<> info etc.
     */
    [[nodiscard]] virtual property_map& metaInformation() noexcept = 0;

    [[nodiscard]] virtual const property_map& metaInformation() const = 0;

    /**
     * @brief used to store non-graph-processing information like UI block position etc.
     */
    [[nodiscard]] virtual property_map& uiConstraints() noexcept = 0;

    [[nodiscard]] virtual const property_map& uiConstraints() const = 0;

    /**
     * @brief process-wide unique name
     * N.B. can be used to disambiguate in case user provided the same 'name()' for several blocks.
     */
    [[nodiscard]] virtual std::string_view uniqueName() const = 0;

    [[nodiscard]] virtual SettingsBase& settings() = 0;

    [[nodiscard]] virtual const SettingsBase& settings() const = 0;

    [[nodiscard]] virtual work::Result work(std::size_t requested_work) = 0;

    [[nodiscard]] virtual work::Status draw(const property_map& config = {}) = 0;

    [[nodiscard]] virtual block::Category blockCategory() const { return block::Category::NormalBlock; }

    virtual void processScheduledMessages() = 0;

    [[nodiscard]] virtual UICategory uiCategory() const { return UICategory::None; }

    // port and sample information
    /**
     * @brief returns the input_chunk_size to output_chunk_size ratio for the block
     */
    [[nodiscard]] virtual gr::Ratio resamplingRatio() const noexcept = 0;

    /**
     * @brief returns the input stride
     */
    [[nodiscard]] virtual gr::Size_t stride() const noexcept = 0;

    /**
     * @brief Bit-mask description of each *input* stream port.
     * @see gr::port::BitMask -> enum class : uint8_t { None = 0U, Input = 1, Stream = 2, Synchronous = 4, Optional = 8, Connected = 16 };
     */
    [[nodiscard]] virtual std::span<const gr::port::BitMask> blockInputTypes() noexcept = 0;

    /**
     * @brief Bit-mask description of each *output* stream port.
     * @see gr::port::BitMask -> enum class : uint8_t { None = 0U, Input = 1, Stream = 2, Synchronous = 4, Optional = 8, Connected = 16 };
     */
    [[nodiscard]] virtual std::span<const gr::port::BitMask> blockOutputTypes() noexcept = 0;

    /**
     * @brief currently available/readable samples per input port.
     * @param reset  if true, forces a fresh read from the ports.
     */
    [[nodiscard]] virtual std::span<const std::size_t> availableInputSamples(bool reset = false) noexcept = 0;

    /**
     * @brief currently available/writeable samples per output port.
     * @param reset  if true, forces a fresh read from the ports.
     */
    [[nodiscard]] virtual std::span<const std::size_t> availableOutputSamples(bool reset = false) noexcept = 0;

    [[nodiscard]] virtual std::span<const std::size_t> minInputRequirements() noexcept  = 0;
    [[nodiscard]] virtual std::span<const std::size_t> maxInputRequirements() noexcept  = 0;
    [[nodiscard]] virtual std::span<const std::size_t> minOutputRequirements() noexcept = 0;
    [[nodiscard]] virtual std::span<const std::size_t> maxOutputRequirements() noexcept = 0;

    [[nodiscard]] virtual bool hasAsyncInputPorts() noexcept  = 0;
    [[nodiscard]] virtual bool hasAsyncOutputPorts() noexcept = 0;

    [[nodiscard]] virtual std::vector<gr::PortMetaInfo> inputMetaInfos(bool reset = true) noexcept  = 0;
    [[nodiscard]] virtual std::vector<gr::PortMetaInfo> outputMetaInfos(bool reset = true) noexcept = 0;

    /**
     * @brief primes/injects dedicated number of samples into input port
     * @param portIdx port index [0, blockInputTypes().size()[
     * @param nSamples number of samples to be primed/injected
     * @return number of samples actually primed (may be < count) or an Error.
     */
    [[nodiscard]] virtual std::expected<std::size_t, gr::Error> primeInputPort(std::size_t portIdx, std::size_t nSamples, std::source_location loc = std::source_location::current()) noexcept = 0;

    [[nodiscard]] virtual void* raw() = 0;
};

namespace detail {
template<typename T, typename... Ts>
constexpr bool contains_type = (std::is_same_v<T, Ts> || ...);
}

template<BlockLike T>
requires std::is_constructible_v<T, property_map>
class BlockWrapper : public BlockModel {
protected:
    static_assert(std::is_same_v<T, std::remove_reference_t<T>>);
    T           _block;
    std::string _type_name = gr::meta::type_name<T>();

    [[nodiscard]] constexpr const auto& blockRef() const noexcept {
        if constexpr (requires { *_block; }) {
            return *_block;
        } else {
            return _block;
        }
    }

    [[nodiscard]] constexpr auto& blockRef() noexcept {
        if constexpr (requires { *_block; }) {
            return *_block;
        } else {
            return _block;
        }
    }

    void initMessagePorts() {
        msgIn  = std::addressof(_block.msgIn);
        msgOut = std::addressof(_block.msgOut);
    }

    void dynamicPortsLoader() {
        if (_dynamicPortsLoaded) {
            return;
        }

        auto processPort = []<typename TPort>(auto& where, TPort& port) -> auto& {
            where.push_back(gr::DynamicPort(port, DynamicPort::non_owned_reference_tag{}));
            return where.back();
        };

        using TBlock = std::remove_cvref_t<decltype(blockRef())>;
        if constexpr (TBlock::blockCategory == block::Category::NormalBlock) {
            auto registerPort = [this, processPort]<gr::detail::PortDescription CurrentPortType>(DynamicPorts& where, auto, CurrentPortType*) noexcept {
                if constexpr (CurrentPortType::kIsDynamicCollection || CurrentPortType::kIsStaticCollection) {
                    auto&               collection = CurrentPortType::getPortObject(blockRef());
                    NamedPortCollection result;
                    result.name = CurrentPortType::Name;
                    for (auto& port : collection) {
                        processPort(result.ports, port);
                    }
                    where.push_back(std::move(result));
                } else {
                    auto& port = CurrentPortType::getPortObject(blockRef());
                    port.name  = CurrentPortType::Name;
                    processPort(where, port);
                }
            };

            traits::block::all_input_ports<TBlock>::for_each(registerPort, _dynamicInputPorts);
            traits::block::all_output_ports<TBlock>::for_each(registerPort, _dynamicOutputPorts);
        }

        _dynamicPortsLoaded = true;
    }

    static void blockWrapperDynamicPortsLoader(BlockModel* base) {
        auto* wrapper = static_cast<BlockWrapper*>(base);
        wrapper->dynamicPortsLoader();
    }

public:
    BlockWrapper() : BlockWrapper(gr::property_map()) {}
    explicit BlockWrapper(gr::property_map initParameter) : _block(std::move(initParameter)) {
        initMessagePorts();
        _dynamicPortsLoader.fn       = &BlockWrapper::blockWrapperDynamicPortsLoader;
        _dynamicPortsLoader.instance = this;
    }

    BlockWrapper(const BlockWrapper& other)            = delete;
    BlockWrapper(BlockWrapper&& other)                 = delete;
    BlockWrapper& operator=(const BlockWrapper& other) = delete;
    BlockWrapper& operator=(BlockWrapper&& other)      = delete;
    ~BlockWrapper() override                           = default;

    void init(std::shared_ptr<gr::Sequence> progress, std::string_view ioThreadPool = gr::thread_pool::kDefaultIoPoolId) override { return blockRef().init(progress, ioThreadPool); }

    [[nodiscard]] constexpr work::Result work(std::size_t requested_work = undefined_size) override { return blockRef().work(requested_work); }

    constexpr work::Status draw(const property_map& config = {}) override {
        if constexpr (requires { blockRef().draw(config); }) {
            return blockRef().draw(config);
        }
        return work::Status::ERROR;
    }

    [[nodiscard]] block::Category blockCategory() const override { return T::blockCategory; }

    [[nodiscard]] UICategory uiCategory() const override { return T::DrawableControl::kCategory; }

    [[nodiscard]] gr::Ratio  resamplingRatio() const noexcept override { return {static_cast<std::int32_t>(blockRef().input_chunk_size), static_cast<std::int32_t>(blockRef().output_chunk_size)}; }
    [[nodiscard]] gr::Size_t stride() const noexcept override { return blockRef().stride; }

    [[nodiscard]] std::span<const port::BitMask> blockInputTypes() noexcept override { return blockRef().inputStreamCache.types(); }
    [[nodiscard]] std::span<const port::BitMask> blockOutputTypes() noexcept override { return blockRef().outputStreamCache.types(); }
    [[nodiscard]] std::span<const std::size_t>   availableInputSamples(bool reset = false) noexcept override { return blockRef().inputStreamCache.availableSamples(reset); }
    [[nodiscard]] std::span<const std::size_t>   availableOutputSamples(bool reset = false) noexcept override { return blockRef().outputStreamCache.availableSamples(reset); }
    [[nodiscard]] std::span<const std::size_t>   minInputRequirements() noexcept override { return blockRef().inputStreamCache.minSamples(); }
    [[nodiscard]] std::span<const std::size_t>   maxInputRequirements() noexcept override { return blockRef().inputStreamCache.maxSamples(); }
    [[nodiscard]] std::span<const std::size_t>   minOutputRequirements() noexcept override { return blockRef().outputStreamCache.minSamples(); }
    [[nodiscard]] std::span<const std::size_t>   maxOutputRequirements() noexcept override { return blockRef().outputStreamCache.maxSamples(); }
    [[nodiscard]] bool                           hasAsyncInputPorts() noexcept override { return blockRef().inputStreamCache.hasASyncAvailable(); }
    [[nodiscard]] bool                           hasAsyncOutputPorts() noexcept override { return blockRef().outputStreamCache.hasASyncAvailable(); }
    [[nodiscard]] std::vector<gr::PortMetaInfo>  inputMetaInfos(bool reset = true) noexcept override { return blockRef().inputStreamCache.metaInfos(reset); }
    [[nodiscard]] std::vector<gr::PortMetaInfo>  outputMetaInfos(bool reset = true) noexcept override { return blockRef().outputStreamCache.metaInfos(reset); }

    [[nodiscard]] std::expected<std::size_t, gr::Error> primeInputPort(std::size_t portIdx, std::size_t nSamples, std::source_location loc = std::source_location::current()) noexcept override { return blockRef().inputStreamCache.primePort(portIdx, nSamples, loc); }

    void processScheduledMessages() override { return blockRef().processScheduledMessages(); }

    // For blocks that contain nested blocks (Graphs, Schedulers)
    [[nodiscard]] std::span<std::shared_ptr<BlockModel>> blocks() noexcept override {
        if constexpr (requires { blockRef().blocks(); }) {
            return blockRef().blocks();
        } else {
            return {};
        }
    }

    [[nodiscard]] std::span<const std::shared_ptr<BlockModel>> blocks() const noexcept override {
        if constexpr (requires { blockRef().blocks(); }) {
            return blockRef().blocks();
        } else {
            return {};
        }
    }

    // For blocks that contain nested blocks (Graphs, Schedulers)
    [[nodiscard]] std::span<Edge> edges() noexcept override {
        if constexpr (requires { blockRef().edges(); }) {
            return blockRef().edges();
        } else {
            return {};
        }
    }

    [[nodiscard]] std::span<const Edge> edges() const noexcept override {
        if constexpr (requires { blockRef().edges(); }) {
            return blockRef().edges();
        } else {
            return {};
        }
    }

    [[nodiscard]] constexpr bool isBlocking() const noexcept override { return blockRef().isBlocking(); }

    [[nodiscard]] std::expected<void, Error> changeStateTo(lifecycle::State newState) noexcept override { return blockRef().changeStateTo(newState); }
    [[nodiscard]] lifecycle::State           state() const noexcept override { return blockRef().state(); }
    [[nodiscard]] std::string_view           name() const override { return blockRef().name; }
    void                                     setName(std::string name) noexcept override { blockRef().name = std::move(name); }
    [[nodiscard]] std::string_view           typeName() const override { return _type_name; }
    [[nodiscard]] property_map&              metaInformation() noexcept override { return blockRef().meta_information; } // TODO: to be removed (read-only)
    [[nodiscard]] const property_map&        metaInformation() const override { return blockRef().meta_information; }
    [[nodiscard]] property_map&              uiConstraints() noexcept override { return blockRef().ui_constraints; }
    [[nodiscard]] const property_map&        uiConstraints() const override { return blockRef().ui_constraints; }
    [[nodiscard]] std::string_view           uniqueName() const override { return blockRef().unique_name; }
    [[nodiscard]] SettingsBase&              settings() override { return blockRef().settings(); }
    [[nodiscard]] const SettingsBase&        settings() const override { return blockRef().settings(); }
    [[nodiscard]] void*                      raw() override { return std::addressof(blockRef()); }
};

} // namespace gr

template<>
struct std::formatter<gr::PortDefinition> {
    constexpr auto parse(std::format_parse_context& ctx) {
        return ctx.begin(); // no format-spec yet
    }

    template<typename FormatContext>
    auto format(const gr::PortDefinition& port, FormatContext& ctx) const {
        if (std::holds_alternative<gr::PortDefinition::IndexBased>(port.definition)) {
            const auto& index = std::get<gr::PortDefinition::IndexBased>(port.definition);
            if (index.subIndex == gr::meta::invalid_index) {
                return std::format_to(ctx.out(), "{}", index.topLevel);
            } else {
                return std::format_to(ctx.out(), "{}#{}", index.topLevel, index.subIndex);
            }
        } else {
            const auto& str = std::get<gr::PortDefinition::StringBased>(port.definition);
            return std::format_to(ctx.out(), "{}", str.name);
        }
    }
};

template<>
struct std::formatter<gr::Edge> {
    char formatSpecifier = 's';

    constexpr auto parse(std::format_parse_context& ctx) {
        auto it = ctx.begin();
        if (it != ctx.end() && (*it == 's' || *it == 'l')) {
            formatSpecifier = *it++;
        } else if (it != ctx.end() && *it != '}') {
            throw std::format_error("invalid format specifier");
        }
        return it;
    }

    template<typename FormatContext>
    auto format(const gr::Edge& e, FormatContext& ctx) const {
        auto getName = [this](const std::shared_ptr<gr::BlockModel>& block) { return (formatSpecifier == 'l') ? block->uniqueName() : block->name(); };

        return std::format_to(ctx.out(), "{}/{} ⟶ (name: '{}', size: {:2}, weight: {:2}, state: {}) ⟶ {}/{}", //
            getName(e._sourceBlock), e._sourcePortDefinition,                                                 // src
            e._name, e._minBufferSize, e._weight, magic_enum::enum_name(e._state),                            // edge
            getName(e._destinationBlock), e._destinationPortDefinition);                                      // dst
    }
};

#endif // GNURADIO_BLOCK_MODEL_HPP
