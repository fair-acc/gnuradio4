#ifndef GNURADIO_PYTHONBLOCK_HPP
#define GNURADIO_PYTHONBLOCK_HPP

#include "PythonInterpreter.hpp"

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/annotated.hpp>

// Forward declaration of PythonBlock method definition, needed for CPython's C-API wrapping
template<typename T>
inline PyModuleDef* myBlockPythonDefinitions(void);

namespace gr::basic {

using namespace gr;

template<typename T>
requires std::is_arithmetic_v<T> /* || std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>> */
struct PythonBlock : public Block<PythonBlock<T>> {
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
std::string pythonScript = R"(
# usual import etc.
counter = 0 # exemplary global state, kept between each invocation

def process_bulk(ins, outs):
    # [..]
    settings = this_block.getSettings()
    print("Current settings:", settings)

    if this_block.tagAvailable(): # tag handling
        tag = this_block.getTag()
        print('Tag:', tag)

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
PythonBlock<int> myBlock(pythonScript); // nominal
myBlock.pythonScript = pythonScript; // alt: only for unit-testing

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
    using A                = Annotated<U, description, Arguments...>;
    using poc_property_map = std::map<std::string, std::string, std::less<>>; // TODO: needs to be replaced with 'property_map` aka. 'pmtv::map_t'
    using tag_type         = std::string;

    std::vector<PortIn<T>>                                                        inputs{};
    std::vector<PortOut<T>>                                                       outputs{};
    A<gr::Size_t, "n_inputs", Visible, Doc<"number of inputs">, Limits<1U, 32U>>  n_inputs     = 0U;
    A<gr::Size_t, "n_outputs", Visible, Doc<"number of inputs">, Limits<1U, 32U>> n_outputs    = 0U;
    std::string                                                                   pythonScript = "";

    PyModuleDef*        _moduleDefinitions = myBlockPythonDefinitions<T>();
    python::Interpreter _interpreter{this, _moduleDefinitions};
    std::string         _prePythonDefinition = fmt::format(R"p(import {0}
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

this_block = PythonBlockWrapper(capsule))p",
                _moduleDefinitions->m_name);
    poc_property_map    _settingsMap{{"key1", "value1"}, {"key2", "value2"}};
    bool                _tagAvailable = false;
    tag_type            _tag          = "Simulated Tag";

    void settingsChanged(const gr::property_map& old_settings, const gr::property_map& new_settings) {
        if (new_settings.contains("n_inputs") || new_settings.contains("n_outputs")) {

            fmt::print("{}: configuration changed: n_inputs {} -> {}, n_outputs {} -> {}\n", this->name, old_settings.at("n_inputs"), new_settings.contains("n_inputs") ? new_settings.at("n_inputs") : "same", old_settings.at("n_outputs"), new_settings.contains("n_outputs") ? new_settings.at("n_outputs") : "same");
            if (std::any_of(inputs.begin(), inputs.end(), [](const auto& port) { return port.isConnected(); })) {
                throw gr::exception("Number of input ports cannot be changed after Graph initialization.");
            }
            if (std::any_of(outputs.begin(), outputs.end(), [](const auto& port) { return port.isConnected(); })) {
                throw gr::exception("Number of output ports cannot be changed after Graph initialization.");
            }
            inputs.resize(n_inputs);
            outputs.resize(n_outputs);
        }

        if (new_settings.contains("pythonScript")) {
            _interpreter.invoke(
                [this] {
                    if (python::PyObjectGuard testImport(PyRun_StringFlags(_prePythonDefinition.data(), Py_file_input, _interpreter.getDictionary(), _interpreter.getDictionary(), nullptr)); !testImport) {
                        python::throwCurrentPythonError(fmt::format("{}(aka. {})::settingsChanged(...) - testImport", this->unique_name, this->name), std::source_location::current(), _prePythonDefinition);
                    }

                    // Retrieve the PythonBlockWrapper class object
                    PyObject* pPythonBlockWrapperClass = PyDict_GetItemString(_interpreter.getDictionary(), "PythonBlockWrapper"); // borrowed reference
                    if (!pPythonBlockWrapperClass) {
                        python::throwCurrentPythonError(fmt::format("{}(aka. {})::settingsChanged(...) - failed to retrieve PythonBlockWrapper class", this->unique_name, this->name), std::source_location::current(), _prePythonDefinition);
                    }

                    // Retrieve the this_block
                    PyObject* pInstance = PyDict_GetItemString(_interpreter.getDictionary(), "this_block"); // borrowed reference
                    if (!pInstance) {
                        python::throwCurrentPythonError(fmt::format("{}(aka. {})::settingsChanged(...) - failed to retrieve 'this_block'", this->unique_name, this->name), std::source_location::current(), _prePythonDefinition);
                    }

                    // Check if pInstance is an instance of PythonBlockWrapper
                    if (PyObject_IsInstance(pInstance, pPythonBlockWrapperClass) != 1) {
                        python::throwCurrentPythonError(fmt::format("{}(aka. {})::settingsChanged(...) - 'this_block' is not an instance of PythonBlockWrapper", this->unique_name, this->name), std::source_location::current(), _prePythonDefinition);
                    }

                    if (const python::PyObjectGuard result(PyRun_StringFlags(pythonScript.data(), Py_file_input, _interpreter.getDictionary(), _interpreter.getDictionary(), nullptr)); !result) {
                        python::throwCurrentPythonError(fmt::format("{}(aka. '{}')::settingsChanged(...) - script parsing error", this->unique_name, this->name), std::source_location::current(), pythonScript);
                    }

                    python::PyObjectGuard pyFunc(PyObject_GetAttrString(_interpreter.getModule(), "process_bulk"));
                    if (!pyFunc.get() || !PyCallable_Check(pyFunc.get())) {
                        python::throwCurrentPythonError(fmt::format("{}(aka. {})::settingsChanged(...) Python function process_bulk not found or is not callable", this->unique_name, this->name), std::source_location::current(), pythonScript);
                    }
                },
                pythonScript);
        }
    }

    const poc_property_map& getSettings() const {
        // TODO: replace with this->settings().get() once the property_map is Python wrapped
        return _settingsMap;
    }

    bool setSettings(const poc_property_map& newSettings) {
        // TODO: replace with this->settings().set(newSettings) once the property_map is Python wrapped
        if (newSettings.empty()) {
            return false;
        }
        for (const auto& [key, value] : newSettings) {
            _settingsMap.insert_or_assign(key, value);
        }
        return true;
    }

    bool tagAvailable() {
        _tagAvailable = !_tagAvailable;
        return _tagAvailable;
    }

    tag_type getTag() { return _tag; }

    template<typename TInputSpan, typename TOutputSpan>
    work::Status processBulk(std::span<TInputSpan> ins, std::span<TOutputSpan> outs) {
        _interpreter.invoke([this, ins, outs] { callPythonFunction(ins, outs); }, pythonScript);
        return work::Status::OK;
    }

    // block life-cycle methods
    // clang-format off
    void start()  { _interpreter.invokeFunction<python::EnforceFunction::OPTIONAL>("start"); }
    void stop()   { _interpreter.invokeFunction<python::EnforceFunction::OPTIONAL>("stop"); }
    void pause()  { _interpreter.invokeFunction<python::EnforceFunction::OPTIONAL>("pause"); }
    void resume() { _interpreter.invokeFunction<python::EnforceFunction::OPTIONAL>("resume"); }
    void reset()  { _interpreter.invokeFunction<python::EnforceFunction::OPTIONAL>("reset"); }
    // clang-format on

private:
    template<typename TInputSpan, typename TOutputSpan>
    void callPythonFunction(std::span<TInputSpan> ins, std::span<TOutputSpan> outs) {
        PyObject* pIns = PyList_New(static_cast<Py_ssize_t>(ins.size()));
        for (size_t i = 0; i < ins.size(); ++i) {
            PyList_SetItem(pIns, Py_ssize_t(i), python::toPyArray(ins[i].data(), {ins[i].size()}));
        }

        PyObject* pOuts = PyList_New(static_cast<Py_ssize_t>(outs.size()));
        for (size_t i = 0; i < outs.size(); ++i) {
            PyList_SetItem(pOuts, Py_ssize_t(i), python::toPyArray(outs[i].data(), {outs[i].size()}));
        }

        python::PyObjectGuard pyArgs(PyTuple_New(2));
        PyTuple_SetItem(pyArgs, 0, pIns);
        PyTuple_SetItem(pyArgs, 1, pOuts);

        if (python::PyObjectGuard pyValue = _interpreter.invokeFunction("process_bulk", pyArgs); !pyValue) {
            python::throwCurrentPythonError(fmt::format("{}(aka. {})::callPythonFunction(..) Python function call failed", this->unique_name, this->name), std::source_location::current(), pythonScript);
        }
    }
};

} // namespace gr::basic
ENABLE_REFLECTION_FOR_TEMPLATE(gr::basic::PythonBlock, inputs, outputs, n_inputs, n_outputs, pythonScript)

template<typename T>
gr::basic::PythonBlock<T>* getPythonBlockFromCapsule(PyObject* capsule) {
    static std::string pyBlockName = gr::python::sanitizedPythonBlockName<gr::basic::PythonBlock<T>>();
    if (void* objPointer = PyCapsule_GetPointer(capsule, pyBlockName.c_str()); objPointer != nullptr) {
        return static_cast<gr::basic::PythonBlock<T>*>(objPointer);
    }
    gr::python::throwCurrentPythonError("could not retrieve obj pointer from capsule");
    return nullptr;
}

template<typename T>
PyObject* PythonBlock_TagAvailable_Template(PyObject* /*self*/, PyObject* args) {
    PyObject* capsule;
    if (!PyArg_ParseTuple(args, "O", &capsule)) {
        return nullptr;
    }
    gr::basic::PythonBlock<T>* myBlock = getPythonBlockFromCapsule<T>(capsule);
    return myBlock->tagAvailable() ? gr::python::TrueObj : gr::python::FalseObj;
}

template<typename T>
PyObject* PythonBlock_GetTag_Template(PyObject* /*self*/, PyObject* args) {
    PyObject* capsule;
    if (!PyArg_ParseTuple(args, "O", &capsule)) {
        return nullptr;
    }
    gr::basic::PythonBlock<T>* myBlock = getPythonBlockFromCapsule<T>(capsule);
    return PyUnicode_FromString(myBlock->getTag().c_str());
}

template<typename T>
PyObject* PythonBlock_GetSettings_Template(PyObject* /*self*/, PyObject* args) {
    PyObject* capsule;
    if (!PyArg_ParseTuple(args, "O", &capsule)) {
        return nullptr;
    }
    const gr::basic::PythonBlock<T>* myBlock = getPythonBlockFromCapsule<T>(capsule);
    if (myBlock == nullptr) {
        gr::python::throwCurrentPythonError(fmt::format("could not retrieve myBLock<{}> {}", gr::meta::type_name<T>(), gr::python::toString(capsule)));
        return nullptr;
    }
    const auto& settings = myBlock->getSettings();

    PyObject* dict = PyDict_New(); // returns owning reference
    if (!dict) {
        return PyErr_NoMemory();
    }
    try {
        for (const auto& [key, value] : settings) {
            gr::python::PyObjectGuard py_value(PyUnicode_FromString(value.c_str()));
            if (!py_value) { // Failed to convert string to Python Unicode
                gr::python::PyDecRef(dict);
                return PyErr_NoMemory();
            }
            // PyDict_SetItemString does not steal reference, so no need to decref py_value
            if (PyDict_SetItemString(dict, key.c_str(), py_value) != 0) {
                gr::python::PyDecRef(dict);
                return nullptr;
            }
        }
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        gr::python::PyDecRef(dict);
        return nullptr;
    }
    return dict;
}

template<typename T>
PyObject* PythonBlock_SetSettings_Template(PyObject* /*self*/, PyObject* args) {
    PyObject* capsule;
    PyObject* settingsDict;
    if (!PyArg_ParseTuple(args, "OO", &capsule, &settingsDict)) {
        return nullptr;
    }
    gr::basic::PythonBlock<T>* myBlock = getPythonBlockFromCapsule<T>(capsule);
    if (!gr::python::isPyDict(settingsDict)) {
        PyErr_SetString(PyExc_TypeError, "Settings must be a dictionary");
        return nullptr;
    }

    typename gr::basic::PythonBlock<T>::poc_property_map newSettings;
    PyObject *                                           key, *value;
    Py_ssize_t                                           pos = 0;
    while (PyDict_Next(settingsDict, &pos, &key, &value)) {
        const char* keyStr   = PyUnicode_AsUTF8(key);
        const char* valueStr = PyUnicode_AsUTF8(value);
        newSettings[keyStr]  = valueStr;
    }

    myBlock->setSettings(newSettings);
    return gr::python::NoneObj;
}

template<typename T>
PyMethodDef* blockMethods() {
    static PyMethodDef methods[] = {
        {"tagAvailable", reinterpret_cast<PyCFunction>(PythonBlock_TagAvailable_Template<T>), METH_VARARGS, "Check if a tag is available"}, {"getTag", reinterpret_cast<PyCFunction>(PythonBlock_GetTag_Template<T>), METH_VARARGS, "Get the current tag"}, {"getSettings", reinterpret_cast<PyCFunction>(PythonBlock_GetSettings_Template<T>), METH_VARARGS, "Get the settings"}, {"setSettings", reinterpret_cast<PyCFunction>(PythonBlock_SetSettings_Template<T>), METH_VARARGS, "Set the settings"}, {nullptr, nullptr, 0, nullptr} // Sentinel
    };
    static_assert(gr::meta::always_false<T>, "type not defined");
    return methods;
}

#define DEFINE_PYTHON_WRAPPER(T, NAME)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         \
    extern "C" inline PyObject* NAME##_##T(PyObject* self, PyObject* args) { return NAME##_Template<T>(self, args); }

#define DEFINE_PYTHON_TYPE_FUNCTIONS_AND_METHODS(type)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         \
    DEFINE_PYTHON_WRAPPER(type, PythonBlock_TagAvailable)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      \
    DEFINE_PYTHON_WRAPPER(type, PythonBlock_GetTag)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            \
    DEFINE_PYTHON_WRAPPER(type, PythonBlock_GetSettings)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       \
    DEFINE_PYTHON_WRAPPER(type, PythonBlock_SetSettings)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       \
    template<>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 \
    PyMethodDef* blockMethods<type>() {                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        \
        static PyMethodDef methods[] = {                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       \
            {"tagAvailable", reinterpret_cast<PyCFunction>(PythonBlock_TagAvailable_##type), METH_VARARGS, "Check if a tag is available"}, {"getTag", reinterpret_cast<PyCFunction>(PythonBlock_GetTag_##type), METH_VARARGS, "Get the current tag"}, {"getSettings", reinterpret_cast<PyCFunction>(PythonBlock_GetSettings_##type), METH_VARARGS, "Get the settings"}, {"setSettings", reinterpret_cast<PyCFunction>(PythonBlock_SetSettings_##type), METH_VARARGS, "Set the settings"}, {nullptr, nullptr, 0, nullptr} /* Sentinel */                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        \
        };                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     \
        return methods;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        \
    }

DEFINE_PYTHON_TYPE_FUNCTIONS_AND_METHODS(int32_t)
DEFINE_PYTHON_TYPE_FUNCTIONS_AND_METHODS(float)

// add more types as needed

template<typename T>
inline PyModuleDef* myBlockPythonDefinitions(void) {
    static std::string  pyBlockName    = gr::python::sanitizedPythonBlockName<gr::basic::PythonBlock<T>>();
    static PyMethodDef* pyBlockMethods = blockMethods<T>();

    constexpr auto            blockDescription = static_cast<std::string_view>(gr::basic::PythonBlock<T>::Description::value);
    static struct PyModuleDef myBlockModule    = {.m_base = PyModuleDef_HEAD_INIT, .m_name = pyBlockName.c_str(), .m_doc = blockDescription.data(), .m_size = -1, .m_methods = pyBlockMethods, .m_slots = nullptr, .m_traverse = nullptr, .m_clear = nullptr, .m_free = nullptr};
    return &myBlockModule;
}

#endif // GNURADIO_PYTHONBLOCK_HPP
