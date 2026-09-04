#ifndef GNURADIO_PYTHONBLOCK_HPP
#define GNURADIO_PYTHONBLOCK_HPP

#include "PythonInterpreter.hpp"

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/annotated.hpp>

// Forward declaration of PythonBlock method definition, needed for CPython's C-API wrapping
template<typename T>
inline PyModuleDef* myBlockPythonDefinitions(void);

namespace gr::basic {

using namespace gr;

GR_REGISTER_BLOCK(gr::basic::PythonBlock, [T], [ int32_t, float ])

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
    using A                 = Annotated<U, description, Arguments...>;
    using StringPropertyMap = std::map<std::string, std::string, std::less<>>; // TODO: replace with gr::property_map once pmt::Value is Python-wrapped
    using tag_type          = std::string;

    std::vector<PortIn<T>>                                                         inputs{};
    std::vector<PortOut<T>>                                                        outputs{};
    A<gr::Size_t, "n_inputs", Visible, Doc<"number of inputs">, Limits<1U, 32U>>   n_inputs     = 1U;
    A<gr::Size_t, "n_outputs", Visible, Doc<"number of outputs">, Limits<1U, 32U>> n_outputs    = 1U;
    std::string                                                                    pythonScript = "";

    GR_MAKE_REFLECTABLE(PythonBlock, inputs, outputs, n_inputs, n_outputs, pythonScript);

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

this_block = PythonBlockWrapper(capsule))p",
                _moduleDefinitions->m_name);
    StringPropertyMap   _settingsMap{{"key1", "value1"}, {"key2", "value2"}};
    bool                _tagAvailable = false;
    tag_type            _tag          = "Simulated Tag";

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

        if (new_settings.contains("pythonScript")) {
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

                    if (const python::PyObjectGuard result(PyRun_StringFlags(pythonScript.data(), Py_file_input, _interpreter.getDictionary(), _interpreter.getDictionary(), nullptr)); !result) {
                        python::throwCurrentPythonError(std::format("{}(aka. '{}')::settingsChanged(...) - script parsing error", this->unique_name, this->name), std::source_location::current(), pythonScript);
                    }

                    python::PyObjectGuard pyFunc(PyObject_GetAttrString(_interpreter.getModule(), "process_bulk"));
                    if (!pyFunc.get() || !PyCallable_Check(pyFunc.get())) {
                        python::throwCurrentPythonError(std::format("{}(aka. {})::settingsChanged(...) Python function process_bulk not found or is not callable", this->unique_name, this->name), std::source_location::current(), pythonScript);
                    }
                },
                pythonScript);
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
    void start()  { invokeLifecycle("start"); }
    void stop()   { invokeLifecycle("stop"); }
    void pause()  { invokeLifecycle("pause"); }
    void resume() { invokeLifecycle("resume"); }
    void reset()  { invokeLifecycle("reset"); }
    // clang-format on

private:
    void invokeLifecycle(std::string_view hook) {
        python::PyGILGuard gil; // PyErr_Occurred() below reads interpreter state
        if (python::PyObjectGuard result = _interpreter.invokeFunction<python::EnforceFunction::OPTIONAL>(hook); !result && PyErr_Occurred()) {
            python::throwCurrentPythonError(std::format("{}(aka. {})::{}() raised", this->unique_name, this->name, hook), std::source_location::current(), pythonScript);
        } // a hook the script does not define yields a null guard with no error set
    }

    template<typename TInputSpan, typename TOutputSpan>
    void callPythonFunction(std::span<TInputSpan> ins, std::span<TOutputSpan> outs) {
        // guarded throughout: toPyArray throws on failure, and the lists must not leak on the way out
        python::PyObjectGuard pyIns(PyList_New(static_cast<Py_ssize_t>(ins.size())));
        python::PyObjectGuard pyOuts(PyList_New(static_cast<Py_ssize_t>(outs.size())));
        python::PyObjectGuard pyArgs(PyTuple_New(2));
        if (!pyIns || !pyOuts || !pyArgs) {
            python::throwCurrentPythonError(std::format("{}(aka. {})::callPythonFunction(..) failed to allocate the Python arguments", this->unique_name, this->name), std::source_location::current(), pythonScript);
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
            python::throwCurrentPythonError(std::format("{}(aka. {})::callPythonFunction(..) Python function call failed", this->unique_name, this->name), std::source_location::current(), pythonScript);
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
        return PyUnicode_FromString(myBlock->getTag().c_str());
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

template<typename T>
inline PyMethodDef* blockMethods() {
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
    inline PyMethodDef* blockMethods<type>() {                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 \
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
