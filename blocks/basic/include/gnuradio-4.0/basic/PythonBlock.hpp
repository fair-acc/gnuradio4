#ifndef GNURADIO_PYTHONBLOCK_HPP
#define GNURADIO_PYTHONBLOCK_HPP

#include "PythonInterpreter.hpp"

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/algorithm/fileio/FileIo.hpp>

#include <array>
#include <cctype>
#include <gnuradio-4.0/annotated.hpp>

// Forward declaration of PythonBlock method definition, needed for CPython's C-API wrapping
template<typename T>
inline PyModuleDef* myBlockPythonDefinitions(void);

namespace gr::basic {

using namespace gr;

GR_REGISTER_BLOCK(gr::basic::PythonBlock, [T], [ int32_t, float, double ])

template<typename T>
requires std::is_arithmetic_v<T> /* || std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>> */
struct PythonBlock : Block<PythonBlock<T>> {
    using Description = Doc<R""(@brief PythonBlock enabling Python scripts to be executed in GR flow-graphs.

This Block encapsulates Python-based scripts that can be executed within the GR flow-graph.
The class manages the Python environment setup, script execution, and data transfer between
C++ and Python. This is a proof-of-concept implementation that can and should be further
extended (e.g. notably pmt-integration, and message handling) but should provide a start for
'processBulk(...)' based signal processing using Python.

Usage Example:
@code
#include <gnuradio-4.0/PythonBlock.hpp>

// [...]
int main() {
// Python script that processes input data arrays and modifies output arrays
std::string python_script = R"(
# usual import etc.
counter = 0 # exemplary global state, kept between each invocation

def process_bulk(ins, outs):
    # [..]
    settings = this_block.getSettings()
    print("Current settings:", settings)

    while this_block.tagAvailable():        # drain the tags on this input span
        tag = this_block.getTag()           # {'index': <sample offset>, 'map': {<key>: <value as str>}}
        print('Tag:', tag['index'], tag['map'])
    this_block.publishTag(0, {'origin': 'python'}) # emit a tag on the outputs

    counter += 1
    # process the input->output samples, here: double each input element
    for i in range(len(ins)):
        outs[i][:] = ins[i] * 2

    # update settings with the counter
    settings["counter"] = str(counter)
    this_block.setSettings(settings)

    # [..]
)";

// C++ side: instantiate PythonBlock with the script
PythonBlock<int> myBlock(python_script); // nominal
myBlock.python_script = python_script; // alt: only for unit-testing

// example for unit-test
std::vector<int>                  data1 = { 1, 2, 3 };
std::vector<int>                  data2 = { 4, 5, 6 };
std::vector<int>                  out1(3); // need std::vector as backing storage
std::vector<int>                  out2(3);
std::vector<std::span<const int>> ins  = { data1, data2 };
std::vector<std::span<int>>       outs = { out1, out2 };

// process data using the Python script
myBlock.processBulk(ins, outs);
// check values of outs
}
@endcode
)"">;
    // optional shortening
    template<typename U, gr::meta::fixed_string description = "", typename... Arguments>
    using A                 = Annotated<U, description, Arguments...>;
    using StringPropertyMap = std::map<std::string, std::string, std::less<>>; // TODO: replace with gr::property_map once pmt::Value is Python-wrapped

    // PoC projection of gr::Tag for the script side: pmt values are rendered as strings until property_map is Python-wrapped
    struct TagView {
        std::size_t       index{0UZ};
        StringPropertyMap map{};
    };

    std::vector<PortIn<T>>                                                                                                                                                                                                                inputs{};
    std::vector<PortOut<T>>                                                                                                                                                                                                               outputs{};
    A<gr::Size_t, "n_inputs", Visible, Doc<"number of inputs">, Limits<1U, 32U>>                                                                                                                                                          n_inputs      = 1U;
    A<gr::Size_t, "n_outputs", Visible, Doc<"number of outputs">, Limits<1U, 32U>>                                                                                                                                                        n_outputs     = 1U;
    A<bool, "forward_tags", Visible, Doc<"copy non-'gr:' input tags to the outputs; the framework forwards the 'gr:' keys regardless">>                                                                                                   forward_tags  = true;
    A<std::string, "python_script", Visible, Doc<"the script itself, or where to fetch it from: a value starting with 'http(s)://', 'file:/', '/', './' or '../' is loaded ('.gz' decoded on the fly), anything else is taken verbatim">> python_script = "";
    A<std::vector<std::string>, "python_path", Doc<"directories prepended to sys.path, so a script can import its own modules">>                                                                                                          python_path   = {};

    GR_MAKE_REFLECTABLE(PythonBlock, inputs, outputs, n_inputs, n_outputs, forward_tags, python_script, python_path);

    PyModuleDef*        _moduleDefinitions = myBlockPythonDefinitions<T>();
    python::Interpreter _interpreter{this, _moduleDefinitions};
    std::string         _prePythonDefinition = std::format(R"p(import {0}
import warnings

class WarningException(Exception):
    """Custom exception class for handling warnings as exceptions with detailed messages."""
    def __init__(self, message, filename=None, lineno=None, category_name=None):
        super().__init__(message)
        self.filename = filename
        self.lineno = lineno
        self.category_name = category_name

def custom_showwarning(message, category, filename, lineno, file=None, line=None):
    raise WarningException(f"{{filename}}:{{lineno}}: {{category.__name__}}: {{message}}", filename=filename, lineno=lineno, category_name=category.__name__) # raise warning as an exception
warnings.showwarning = custom_showwarning
warnings.simplefilter('always') # trigger on all warnings, can be adjusted as needed

class PythonBlockWrapper: ## helper class to make the C++ class appear as a Python class
    def __init__(self, capsule):
        self.capsule = capsule
    def tagAvailable(self):
        return {0}.tagAvailable(self.capsule)
    def getTag(self):
        return {0}.getTag(self.capsule)
    def getSettings(self):
        return {0}.getSettings(self.capsule)
    def setSettings(self, settings):
        {0}.setSettings(self.capsule, settings)
    def publishTag(self, offset, tagData):
        {0}.publishTag(self.capsule, offset, tagData)
    def log(self, message):
        {0}.log(self.capsule, message)

this_block = PythonBlockWrapper(capsule)

def print(*args, sep=' ', end='\n', file=None, flush=False): ## shadows the builtin for THIS block's module only, so script output joins the GR4 log
    this_block.log(sep.join(str(arg) for arg in args)))p",
                _moduleDefinitions->m_name);
    std::string         _activeScript{};    // python_script verbatim, or the contents it points at
    std::string         _scriptLoadError{}; // init() records a settingsChanged throw as an error state, so processBulk reports the reason
    StringPropertyMap   _settingsMap{{"key1", "value1"}, {"key2", "value2"}};

    std::vector<TagView>                                  _inputTags{};         // string projection handed to the script
    std::size_t                                           _nextInputTag = 0UZ;  // read cursor consumed by getTag()
    std::vector<std::pair<std::size_t, gr::property_map>> _forwardTags{};       // untouched originals, so forwarding keeps the pmt types
    std::vector<std::pair<std::size_t, gr::property_map>> _pendingOutputTags{}; // queued by the script, published once it returns

    // block life-cycle methods
    // clang-format off
    void start()  { invokeLifecycle("start"); }
    void stop()   { invokeLifecycle("stop"); }
    void pause()  { invokeLifecycle("pause"); }
    void resume() { invokeLifecycle("resume"); }
    void reset()  { invokeLifecycle("reset"); }
    // clang-format on

    template<typename TInputSpan, typename TOutputSpan>
    work::Status processBulk(std::span<TInputSpan> ins, std::span<TOutputSpan> outs) {
        if (!_scriptLoadError.empty()) { // no script is loaded, so report why rather than let process_bulk turn up missing
            throw gr::exception(_scriptLoadError);
        }
        collectInputTags(ins);
        _interpreter.invoke([this, ins, outs] { callPythonFunction(ins, outs); }, _activeScript);
        publishPendingTags(outs);
        return work::Status::OK;
    }

    void settingsChanged(const gr::property_map& /*old_settings*/, const gr::property_map& new_settings) {
        if (inputs.size() != n_inputs || outputs.size() != n_outputs) { // drive off the actual port count, so the defaults are applied too
            gr::log::debug("{}: port configuration changed: n_inputs {} -> {}, n_outputs {} -> {}", this->name, inputs.size(), n_inputs, outputs.size(), n_outputs);
            if (std::any_of(inputs.begin(), inputs.end(), [](const auto& port) { return port.isConnected(); })) {
                throw gr::exception("Number of input ports cannot be changed after Graph initialization.");
            }
            if (std::any_of(outputs.begin(), outputs.end(), [](const auto& port) { return port.isConnected(); })) {
                throw gr::exception("Number of output ports cannot be changed after Graph initialization.");
            }
            inputs.resize(n_inputs);
            outputs.resize(n_outputs);
        }

        if (new_settings.contains("python_script") || new_settings.contains("python_path")) {
            _scriptLoadError = std::format("{}: script configuration did not complete", this->name); // replaced by the actual reason, cleared once valid
            _activeScript    = loadScript();
            try {
                _interpreter.invoke(
                    [this] {
                        if (python::PyObjectGuard testImport(PyRun_StringFlags(_prePythonDefinition.data(), Py_file_input, _interpreter.getDictionary(), _interpreter.getDictionary(), nullptr)); !testImport) {
                            python::throwCurrentPythonError(std::format("{}(aka. {})::settingsChanged(...) - testImport", this->unique_name, this->name), std::source_location::current(), _prePythonDefinition);
                        }

                        // Retrieve the PythonBlockWrapper class object
                        PyObject* pPythonBlockWrapperClass = PyDict_GetItemString(_interpreter.getDictionary(), "PythonBlockWrapper"); // borrowed reference
                        if (!pPythonBlockWrapperClass) {
                            python::throwCurrentPythonError(std::format("{}(aka. {})::settingsChanged(...) - failed to retrieve PythonBlockWrapper class", this->unique_name, this->name), std::source_location::current(), _prePythonDefinition);
                        }

                        // Retrieve the this_block
                        PyObject* pInstance = PyDict_GetItemString(_interpreter.getDictionary(), "this_block"); // borrowed reference
                        if (!pInstance) {
                            python::throwCurrentPythonError(std::format("{}(aka. {})::settingsChanged(...) - failed to retrieve 'this_block'", this->unique_name, this->name), std::source_location::current(), _prePythonDefinition);
                        }

                        // Check if pInstance is an instance of PythonBlockWrapper
                        if (PyObject_IsInstance(pInstance, pPythonBlockWrapperClass) != 1) {
                            python::throwCurrentPythonError(std::format("{}(aka. {})::settingsChanged(...) - 'this_block' is not an instance of PythonBlockWrapper", this->unique_name, this->name), std::source_location::current(), _prePythonDefinition);
                        }

                        prependPythonPath();
                        if (const python::PyObjectGuard result(PyRun_StringFlags(_activeScript.c_str(), Py_file_input, _interpreter.getDictionary(), _interpreter.getDictionary(), nullptr)); !result) {
                            python::throwCurrentPythonError(std::format("{}(aka. '{}')::settingsChanged(...) - script parsing error", this->unique_name, this->name), std::source_location::current(), _activeScript);
                        }

                        python::PyObjectGuard pyFunc(PyObject_GetAttrString(_interpreter.getModule(), "process_bulk"));
                        if (!pyFunc.get() || !PyCallable_Check(pyFunc.get())) {
                            python::throwCurrentPythonError(std::format("{}(aka. {})::settingsChanged(...) Python function process_bulk not found or is not callable", this->unique_name, this->name), std::source_location::current(), _activeScript);
                        }
                        _scriptLoadError.clear(); // the script parsed and exposes process_bulk
                    },
                    _activeScript);
            } catch (const std::exception& configurationError) {
                _scriptLoadError = configurationError.what(); // init() only records the throw, so keep the reason for processBulk
                throw;
            }
        }
    }

    const StringPropertyMap& getSettings() const {
        // TODO: replace with this->settings().get() once the property_map is Python wrapped
        return _settingsMap;
    }

    bool setSettings(const StringPropertyMap& newSettings) {
        // TODO: replace with this->settings().set(newSettings) once the property_map is Python wrapped
        if (newSettings.empty()) {
            return false;
        }
        for (const auto& [key, value] : newSettings) {
            _settingsMap.insert_or_assign(key, value);
        }
        return true;
    }

    [[nodiscard]] bool tagAvailable() const { return _nextInputTag < _inputTags.size(); }

    TagView getTag() { // consumes one tag, so a script can drain them with `while this_block.tagAvailable():`
        if (_nextInputTag >= _inputTags.size()) {
            return {};
        }
        return _inputTags[_nextInputTag++];
    }

    // The complete set of prefixes that mark a location; anything else is the script itself. Every entry ends in '/'
    // and matching is anchored, so an annotated global ("rate: float = 1e6") and a script mentioning a URL stay verbatim.
    // N.B. limited to what readAsync() serves on every platform: 'download:/' is write-only, 'dialog:/' Emscripten-only.
    static constexpr std::array<std::string_view, 3> kLocationSchemes{"http://", "https://", "file:/"};
    static constexpr std::array<std::string_view, 3> kLocationPaths{"/", "./", "../"}; // no Python file may begin with these

    [[nodiscard]] static std::string_view trimmed(std::string_view value) noexcept {
        constexpr std::string_view kSpace = " \t\r\n";
        const auto                 first  = value.find_first_not_of(kSpace);
        if (first == std::string_view::npos) {
            return {};
        }
        return value.substr(first, value.find_last_not_of(kSpace) - first + 1UZ);
    }

    [[nodiscard]] static bool isScriptLocation(std::string_view value) noexcept {
        value                    = trimmed(value);
        const auto matchesScheme = [value](std::string_view scheme) noexcept { // URI schemes are case-insensitive
            return value.size() >= scheme.size() && std::ranges::equal(value.substr(0UZ, scheme.size()), scheme, [](char lhs, char rhs) noexcept { return std::tolower(static_cast<unsigned char>(lhs)) == static_cast<unsigned char>(rhs); });
        };
        return std::ranges::any_of(kLocationSchemes, matchesScheme) || std::ranges::any_of(kLocationPaths, [value](std::string_view prefix) noexcept { return value.starts_with(prefix); });
    }

    void log(std::string_view message) const { gr::log::info("{}: {}", this->name, message); }

    void publishTag(std::size_t offset, const StringPropertyMap& tagData) {
        gr::property_map map;
        for (const auto& [key, value] : tagData) {
            map[key] = value; // PoC: every value travels as a string
        }
        _pendingOutputTags.emplace_back(offset, std::move(map));
    }

private:
    void invokeLifecycle(std::string_view hook) {
        python::PyGILGuard gil; // PyErr_Occurred() below reads interpreter state
        if (python::PyObjectGuard result = _interpreter.invokeFunction<python::EnforceFunction::OPTIONAL>(hook); !result && PyErr_Occurred()) {
            python::throwCurrentPythonError(std::format("{}(aka. {})::{}() raised", this->unique_name, this->name, hook), std::source_location::current(), _activeScript);
        } // a hook the script does not define yields a null guard with no error set
    }

    // std::format renders a pmt string with surrounding quotes; the script must see the text itself
    [[nodiscard]] static std::string toScriptText(const auto& value) {
        if (value.value_type() == gr::pmt::Value::ValueType::String) {
            return value.value_or(std::string{});
        }
        return std::format("{}", value);
    }

    [[nodiscard]] std::string loadScript() {
        if (!isScriptLocation(python_script)) {
            return python_script;
        }
        const std::string location(trimmed(python_script));
        auto              reader = gr::algorithm::fileio::readAsync(location);
        if (!reader) {
            _scriptLoadError = std::format("{}: cannot open python_script location '{}': {}", this->name, location, reader.error().message);
            throw gr::exception(_scriptLoadError);
        }
        auto data = reader->get();
        if (!data) {
            _scriptLoadError = std::format("{}: cannot read python_script location '{}': {}", this->name, location, data.error().message);
            throw gr::exception(_scriptLoadError);
        }
        return std::string(reinterpret_cast<const char*>(data->data()), data->size());
    }

    void prependPythonPath() const { // N.B. sys.path is process-global, so entries are shared with every other block
        if (python_path.value.empty()) {
            return;
        }
        PyObject* sysPath = PySys_GetObject("path"); // borrowed reference
        if (sysPath == nullptr) {
            python::throwCurrentPythonError(std::format("{}: sys.path is unavailable", this->name));
        }
        for (const std::string& directory : python_path.value | std::views::reverse) { // each entry goes to index 0, so walk backwards to preserve the given order
            python::PyObjectGuard entry(PyUnicode_FromString(directory.c_str()));
            if (!entry) {
                python::throwCurrentPythonError(std::format("{}: cannot convert python_path entry '{}'", this->name, directory));
            }
            if (PySequence_Contains(sysPath, entry) == 1) { // re-configuration must not grow sys.path without bound
                continue;
            }
            if (PyList_Insert(sysPath, 0, entry) != 0) {
                python::throwCurrentPythonError(std::format("{}: cannot prepend '{}' to sys.path", this->name, directory));
            }
        }
    }

    // N.B. the `requires` guards keep the block usable with plain std::span, which the unit tests and the usage example pass directly
    template<typename TInputSpan>
    void collectInputTags(std::span<TInputSpan> ins) {
        _inputTags.clear();
        _forwardTags.clear();
        _nextInputTag = 0UZ;
        _pendingOutputTags.clear(); // a throwing script must not leak its tags into the next invocation
        // N.B. tags() is relative to the start of this span, which is exactly what publishTag() expects; rawTags() would be absolute
        if constexpr (requires(TInputSpan span) { span.tags(); }) {
            for (TInputSpan& span : ins) {
                for (const auto& [relativeIndex, mapRef] : span.tags()) {
                    const std::size_t offset = relativeIndex < 0 ? 0UZ : static_cast<std::size_t>(relativeIndex); // a tag of already-consumed samples belongs at the chunk start
                    TagView           view{offset, {}};
                    gr::property_map  forwarded;
                    for (const auto& [key, value] : mapRef.get()) {
                        view.map[std::string(key)] = toScriptText(value);
                        if (!std::string_view(key).starts_with(gr::GR_TAG_PREFIX.view())) {
                            forwarded[key] = value; // Block<>::forwardInputTags already carries the 'gr:' keys
                        }
                    }
                    _inputTags.push_back(std::move(view));
                    const auto duplicate = [&](const auto& entry) { return entry.first == offset && entry.second == forwarded; };
                    if (!forwarded.empty() && std::ranges::none_of(_forwardTags, duplicate)) { // one tag on several inputs is still one tag
                        _forwardTags.emplace_back(offset, std::move(forwarded));
                    }
                }
            }
        }
    }

    template<typename TOutputSpan>
    void publishPendingTags(std::span<TOutputSpan> outs) {
        if constexpr (requires(TOutputSpan span, gr::property_map map) { span.publishTag(map, 0UZ); }) {
            std::vector<std::pair<std::size_t, gr::property_map>> outgoing;
            if (forward_tags) {
                outgoing = _forwardTags;
            }
            outgoing.insert(outgoing.end(), _pendingOutputTags.begin(), _pendingOutputTags.end());
            std::ranges::stable_sort(outgoing, {}, &std::pair<std::size_t, gr::property_map>::first); // publishTag() aborts on a descending index

            for (TOutputSpan& span : outs) {
                for (const auto& [offset, map] : outgoing) {
                    if (offset >= span.size()) { // a tag beyond the chunk would land on samples this call does not produce
                        gr::log::warning("{}: dropping tag at offset {}, outside a span of {} samples", this->name, offset, span.size());
                        continue;
                    }
                    span.publishTag(map, offset);
                }
            }
        }
        _pendingOutputTags.clear();
    }

    template<typename TInputSpan, typename TOutputSpan>
    void callPythonFunction(std::span<TInputSpan> ins, std::span<TOutputSpan> outs) {
        // guarded throughout: toPyArray throws on failure, and the lists must not leak on the way out
        python::PyObjectGuard pyIns(PyList_New(static_cast<Py_ssize_t>(ins.size())));
        python::PyObjectGuard pyOuts(PyList_New(static_cast<Py_ssize_t>(outs.size())));
        python::PyObjectGuard pyArgs(PyTuple_New(2));
        if (!pyIns || !pyOuts || !pyArgs) {
            python::throwCurrentPythonError(std::format("{}(aka. {})::callPythonFunction(..) failed to allocate the Python arguments", this->unique_name, this->name), std::source_location::current(), _activeScript);
        }

        for (std::size_t i = 0; i < ins.size(); ++i) {
            PyList_SetItem(pyIns, static_cast<Py_ssize_t>(i), python::toPyArray(ins[i].data(), {ins[i].size()})); // steals the array reference
        }
        for (std::size_t i = 0; i < outs.size(); ++i) {
            PyList_SetItem(pyOuts, static_cast<Py_ssize_t>(i), python::toPyArray(outs[i].data(), {outs[i].size()}));
        }

        python::PyIncRef(pyIns); // PyTuple_SetItem steals a reference that the guard still holds
        PyTuple_SetItem(pyArgs, 0, pyIns);
        python::PyIncRef(pyOuts);
        PyTuple_SetItem(pyArgs, 1, pyOuts);

        if (python::PyObjectGuard pyValue = _interpreter.invokeFunction("process_bulk", pyArgs); !pyValue) {
            python::throwCurrentPythonError(std::format("{}(aka. {})::callPythonFunction(..) Python function call failed", this->unique_name, this->name), std::source_location::current(), _activeScript);
        }
    }
};

} // namespace gr::basic

// returns nullptr and leaves the Python error set by PyCapsule_GetPointer; callers are CPython callbacks and must not throw
template<typename T>
gr::basic::PythonBlock<T>* getPythonBlockFromCapsule(PyObject* capsule) {
    static std::string pyBlockName = gr::python::sanitizedPythonBlockName<gr::basic::PythonBlock<T>>();
    return static_cast<gr::basic::PythonBlock<T>*>(PyCapsule_GetPointer(capsule, pyBlockName.c_str()));
}

template<typename T>
PyObject* PythonBlock_TagAvailable_Template(PyObject* /*self*/, PyObject* args) {
    return gr::python::invokeCallback([args]() -> PyObject* {
        PyObject* capsule = nullptr;
        if (!PyArg_ParseTuple(args, "O", &capsule)) {
            return nullptr;
        }
        gr::basic::PythonBlock<T>* myBlock = getPythonBlockFromCapsule<T>(capsule);
        if (myBlock == nullptr) {
            return nullptr;
        }
        PyObject* available = myBlock->tagAvailable() ? gr::python::TrueObj : gr::python::FalseObj;
        gr::python::PyIncRef(available); // a callback must return a new reference, not a borrowed one
        return available;
    });
}

template<typename T>
PyObject* PythonBlock_GetTag_Template(PyObject* /*self*/, PyObject* args) {
    return gr::python::invokeCallback([args]() -> PyObject* {
        PyObject* capsule = nullptr;
        if (!PyArg_ParseTuple(args, "O", &capsule)) {
            return nullptr;
        }
        gr::basic::PythonBlock<T>* myBlock = getPythonBlockFromCapsule<T>(capsule);
        if (myBlock == nullptr) {
            return nullptr;
        }

        const auto                tag = myBlock->getTag();
        gr::python::PyObjectGuard tagMap(PyDict_New());
        if (!tagMap) {
            return PyErr_NoMemory();
        }
        for (const auto& [key, value] : tag.map) {
            gr::python::PyObjectGuard pyValue(PyUnicode_FromString(value.c_str()));
            if (!pyValue || PyDict_SetItemString(tagMap, key.c_str(), pyValue) != 0) {
                return nullptr;
            }
        }

        gr::python::PyObjectGuard result(PyDict_New());
        gr::python::PyObjectGuard pyIndex(PyLong_FromSize_t(tag.index));
        if (!result || !pyIndex) {
            return PyErr_NoMemory();
        }
        if (PyDict_SetItemString(result, "index", pyIndex) != 0 || PyDict_SetItemString(result, "map", tagMap) != 0) {
            return nullptr;
        }
        gr::python::PyIncRef(result); // hand the caller its own reference, the guard releases ours
        return result.get();
    });
}

template<typename T>
PyObject* PythonBlock_GetSettings_Template(PyObject* /*self*/, PyObject* args) {
    return gr::python::invokeCallback([args]() -> PyObject* {
        PyObject* capsule = nullptr;
        if (!PyArg_ParseTuple(args, "O", &capsule)) {
            return nullptr;
        }
        const gr::basic::PythonBlock<T>* myBlock = getPythonBlockFromCapsule<T>(capsule);
        if (myBlock == nullptr) {
            return nullptr;
        }

        gr::python::PyObjectGuard dict(PyDict_New());
        if (!dict) {
            return PyErr_NoMemory();
        }
        for (const auto& [key, value] : myBlock->getSettings()) {
            gr::python::PyObjectGuard pyValue(PyUnicode_FromString(value.c_str()));
            if (!pyValue) {
                return PyErr_NoMemory();
            }
            if (PyDict_SetItemString(dict, key.c_str(), pyValue) != 0) { // does not steal the reference
                return nullptr;
            }
        }
        gr::python::PyIncRef(dict); // hand the caller its own reference, the guard releases ours
        return dict.get();
    });
}

template<typename T>
PyObject* PythonBlock_SetSettings_Template(PyObject* /*self*/, PyObject* args) {
    return gr::python::invokeCallback([args]() -> PyObject* {
        PyObject* capsule      = nullptr;
        PyObject* settingsDict = nullptr;
        if (!PyArg_ParseTuple(args, "OO", &capsule, &settingsDict)) {
            return nullptr;
        }
        gr::basic::PythonBlock<T>* myBlock = getPythonBlockFromCapsule<T>(capsule);
        if (myBlock == nullptr) {
            return nullptr;
        }
        if (!gr::python::isPyDict(settingsDict)) {
            PyErr_SetString(PyExc_TypeError, "settings must be a dictionary");
            return nullptr;
        }

        typename gr::basic::PythonBlock<T>::StringPropertyMap newSettings;
        PyObject*                                             key   = nullptr;
        PyObject*                                             value = nullptr;
        Py_ssize_t                                            pos   = 0;
        while (PyDict_Next(settingsDict, &pos, &key, &value)) {
            const char* keyStr   = PyUnicode_AsUTF8(key);
            const char* valueStr = PyUnicode_AsUTF8(value);
            if (keyStr == nullptr || valueStr == nullptr) { // non-str key or value -- PyUnicode_AsUTF8 returned nullptr
                PyErr_SetString(PyExc_TypeError, "settings keys and values must be strings");
                return nullptr;
            }
            newSettings[keyStr] = valueStr;
        }

        myBlock->setSettings(newSettings);
        gr::python::PyIncRef(gr::python::NoneObj); // a callback must return a new reference, not a borrowed one
        return gr::python::NoneObj;
    });
}

// only the DEFINE_PYTHON_TYPE_FUNCTIONS_AND_METHODS(type) specialisations below are usable; instantiating the primary template is the diagnostic
template<typename T>
PyObject* PythonBlock_Log_Template(PyObject* /*self*/, PyObject* args) {
    return gr::python::invokeCallback([args]() -> PyObject* {
        PyObject*   capsule = nullptr;
        const char* message = nullptr;
        if (!PyArg_ParseTuple(args, "Os", &capsule, &message)) {
            return nullptr;
        }
        gr::basic::PythonBlock<T>* myBlock = getPythonBlockFromCapsule<T>(capsule);
        if (myBlock == nullptr) {
            return nullptr;
        }
        myBlock->log(message);
        gr::python::PyIncRef(gr::python::NoneObj);
        return gr::python::NoneObj;
    });
}

template<typename T>
PyObject* PythonBlock_PublishTag_Template(PyObject* /*self*/, PyObject* args) {
    return gr::python::invokeCallback([args]() -> PyObject* {
        PyObject*  capsule = nullptr;
        PyObject*  tagData = nullptr;
        Py_ssize_t offset  = 0;
        if (!PyArg_ParseTuple(args, "OnO", &capsule, &offset, &tagData)) {
            return nullptr;
        }
        gr::basic::PythonBlock<T>* myBlock = getPythonBlockFromCapsule<T>(capsule);
        if (myBlock == nullptr) {
            return nullptr;
        }
        if (offset < 0) {
            PyErr_SetString(PyExc_ValueError, "tag offset must not be negative");
            return nullptr;
        }
        if (!gr::python::isPyDict(tagData)) {
            PyErr_SetString(PyExc_TypeError, "tag data must be a dictionary");
            return nullptr;
        }

        typename gr::basic::PythonBlock<T>::StringPropertyMap tagMap;
        PyObject*                                             key   = nullptr;
        PyObject*                                             value = nullptr;
        Py_ssize_t                                            pos   = 0;
        while (PyDict_Next(tagData, &pos, &key, &value)) {
            const char* keyStr   = PyUnicode_AsUTF8(key);
            const char* valueStr = PyUnicode_AsUTF8(value);
            if (keyStr == nullptr || valueStr == nullptr) {
                PyErr_SetString(PyExc_TypeError, "tag keys and values must be strings");
                return nullptr;
            }
            tagMap[keyStr] = valueStr;
        }

        myBlock->publishTag(static_cast<std::size_t>(offset), tagMap);
        gr::python::PyIncRef(gr::python::NoneObj);
        return gr::python::NoneObj;
    });
}

template<typename T>
inline PyMethodDef* blockMethods() {
    static_assert(gr::meta::always_false<T>, "PythonBlock<T>: no Python method table for this T -- add DEFINE_PYTHON_TYPE_FUNCTIONS_AND_METHODS(T) and register T with GR_REGISTER_BLOCK");
    return nullptr;
}

#define DEFINE_PYTHON_WRAPPER(T, NAME)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         \
    extern "C" inline PyObject* NAME##_##T(PyObject* self, PyObject* args) { return NAME##_Template<T>(self, args); }

#define DEFINE_PYTHON_TYPE_FUNCTIONS_AND_METHODS(type)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         \
    DEFINE_PYTHON_WRAPPER(type, PythonBlock_TagAvailable)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      \
    DEFINE_PYTHON_WRAPPER(type, PythonBlock_GetTag)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            \
    DEFINE_PYTHON_WRAPPER(type, PythonBlock_GetSettings)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       \
    DEFINE_PYTHON_WRAPPER(type, PythonBlock_SetSettings)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       \
    DEFINE_PYTHON_WRAPPER(type, PythonBlock_PublishTag)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        \
    DEFINE_PYTHON_WRAPPER(type, PythonBlock_Log)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               \
    template<>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 \
    inline PyMethodDef* blockMethods<type>() {                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 \
        static PyMethodDef methods[] = {                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       \
            {"tagAvailable", reinterpret_cast<PyCFunction>(PythonBlock_TagAvailable_##type), METH_VARARGS, "Check if a tag is available"}, {"getTag", reinterpret_cast<PyCFunction>(PythonBlock_GetTag_##type), METH_VARARGS, "Get the current tag"}, {"getSettings", reinterpret_cast<PyCFunction>(PythonBlock_GetSettings_##type), METH_VARARGS, "Get the settings"}, {"setSettings", reinterpret_cast<PyCFunction>(PythonBlock_SetSettings_##type), METH_VARARGS, "Set the settings"}, {"publishTag", reinterpret_cast<PyCFunction>(PythonBlock_PublishTag_##type), METH_VARARGS, "Publish a tag on the outputs"}, {"log", reinterpret_cast<PyCFunction>(PythonBlock_Log_##type), METH_VARARGS, "Write a message to the GR4 log"}, {nullptr, nullptr, 0, nullptr} /* Sentinel */                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            \
        };                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     \
        return methods;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        \
    }

DEFINE_PYTHON_TYPE_FUNCTIONS_AND_METHODS(int32_t)
DEFINE_PYTHON_TYPE_FUNCTIONS_AND_METHODS(float)
DEFINE_PYTHON_TYPE_FUNCTIONS_AND_METHODS(double)

// add more types as needed

template<typename T>
inline PyModuleDef* myBlockPythonDefinitions(void) {
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
#ifndef __clang__
#pragma GCC diagnostic ignored "-Wuseless-cast"
#endif
#endif
    static std::string  pyBlockName    = gr::python::sanitizedPythonBlockName<gr::basic::PythonBlock<T>>();
    static PyMethodDef* pyBlockMethods = blockMethods<T>();

    constexpr auto            blockDescription = static_cast<std::string_view>(gr::basic::PythonBlock<T>::Description::value);
    static struct PyModuleDef myBlockModule    = {.m_base = PyModuleDef_HEAD_INIT, .m_name = pyBlockName.c_str(), .m_doc = blockDescription.data(), .m_size = -1, .m_methods = pyBlockMethods, .m_slots = nullptr, .m_traverse = nullptr, .m_clear = nullptr, .m_free = nullptr};
    return &myBlockModule;
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
}

#endif // GNURADIO_PYTHONBLOCK_HPP
