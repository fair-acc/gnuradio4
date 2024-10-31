#ifndef GNURADIO_PYTHONINTERPRETER_HPP
#define GNURADIO_PYTHONINTERPRETER_HPP

#include <atomic>
#include <cassert>
#include <cctype>
#include <complex>
#include <cstdint>
#include <regex>
#include <span>
#include <sstream>
#include <string>
#include <string_view>

#include <gnuradio-4.0/Message.hpp>

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
#ifndef __clang__
#pragma GCC diagnostic ignored "-Wuseless-cast"
#endif
#endif
#include <Python.h>

#include <numpy/numpyconfig.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

namespace gr::python {

inline static PyObject* TrueObj  = Py_True;
inline static PyObject* FalseObj = Py_False;
inline static PyObject* NoneObj  = Py_None;

constexpr inline bool isPyDict(const PyObject* obj) { return PyDict_Check(obj); }

constexpr inline void PyDecRef(PyObject* obj) { // wrapper to isolate unsafe warning on C-API casts
    Py_XDECREF(obj);
}

constexpr inline void PyIncRef(PyObject* obj) { // wrapper to isolate unsafe warning on C-API casts
    Py_XINCREF(obj);
}

constexpr inline std::string PyBytesAsString(PyObject* op) { return PyBytes_AsString(op); }

class PyObjectGuard {
    PyObject* _ptr;

    void move(PyObjectGuard&& other) noexcept {
        PyDecRef(_ptr);
        std::swap(_ptr, other._ptr);
    }

public:
    explicit PyObjectGuard(PyObject* ptr = nullptr) : _ptr(ptr) {}

    explicit PyObjectGuard(PyObjectGuard&& other) noexcept : _ptr(other._ptr) { move(std::move(other)); }

    ~PyObjectGuard() { PyDecRef(_ptr); }

    PyObjectGuard& operator=(PyObjectGuard&& other) noexcept {
        if (this != &other) {
            move(std::move(other));
        }
        return *this;
    }

    PyObjectGuard(const PyObjectGuard& other) : _ptr(other._ptr) { // copy constructor
        PyIncRef(_ptr);
    }

    PyObjectGuard& operator=(const PyObjectGuard& other) {
        if (this == &other) {
            return *this;
        }
        _ptr = other._ptr;
        PyIncRef(_ptr);
        return *this;
    }

    operator PyObject*() const { return _ptr; }

    PyObject* get() const { return _ptr; }
};

class PyGILGuard {
    PyGILState_STATE _state;

public:
    PyGILGuard() : _state(PyGILState_Ensure()) {}

    ~PyGILGuard() { PyGILState_Release(_state); }

    PyGILGuard(const PyGILGuard&)            = delete;
    PyGILGuard& operator=(const PyGILGuard&) = delete;
};

[[nodiscard]] inline std::string toString(PyObject* object) {
    PyObjectGuard strObj(PyObject_Repr(object));
    PyObjectGuard bytesObj(PyUnicode_AsEncodedString(strObj.get(), "utf-8", "strict"));
    return python::PyBytesAsString(bytesObj.get());
}

[[nodiscard]] inline std::string toLineCountAnnotated(std::string_view code, std::size_t min = 0UZ, std::size_t max = std::numeric_limits<std::size_t>::max(), std::size_t marker = std::numeric_limits<std::size_t>::max() - 1UZ) {
    if (code.empty()) {
        return "";
    }
    auto splitLines = [](std::string_view str) {
        std::istringstream       stream{std::string(str)}; // Convert string_view to string
        std::vector<std::string> lines;
        std::string              line;
        while (std::getline(stream, line)) {
            lines.push_back(line);
        }
        return lines;
    };

    auto        lines = splitLines(code);
    std::string annotatedCode;
    annotatedCode.reserve(code.size() + lines.size() * 4UZ /*sizeof "123:"*/);
    for (std::size_t i = std::max(0UZ, min); i < std::min(lines.size(), max); i++) {
        // N.B. Python starts counting from '1' not '0'
        annotatedCode += std::format("{:3}:{}{}\n", i, lines[i], i == marker - 1UZ ? "   ####### <== here's your problem #######" : "");
    }
    return annotatedCode;
}

[[nodiscard]] inline std::string getDebugPythonObjectAttributes(PyObject* obj) {
    if (obj == nullptr) {
        return "The provided PyObject is null.\n";
    }

    PyObjectGuard dirList(PyObject_Dir(obj));
    if (!dirList) {
        PyErr_Print();
        return "Failed to get attribute list from object.\n";
    }

    // iterate over the list of attribute names
    std::string ret;
    Py_ssize_t  size = PyList_Size(dirList);
    for (Py_ssize_t i = 0; i < size; i++) {
        PyObject*     attrName = PyList_GetItem(dirList, i); // borrowed reference, no need to decref
        PyObjectGuard attrValue(PyObject_GetAttr(obj, attrName));
        ret += std::format("item {:3}: key: {} value: {}\n", i, toString(attrName), attrValue ? toString(attrValue) : "<Unable to retrieve value>");
    }
    return ret;
}

inline void throwCurrentPythonError(std::string_view msg, std::source_location location = std::source_location::current(), std::string_view pythonCode = "") {
    PyObjectGuard exception(PyErr_GetRaisedException());
    if (!exception) {
        throw gr::exception(std::format("{}\nPython error: <unknown exception>\ntrace-back: {}", msg, toLineCountAnnotated(pythonCode)), location);
    }
    // fmt::println("detailed debug info: {}", getDebugPythonObjectAttributes(exception))

    std::size_t min    = 0UZ;
    std::size_t max    = std::numeric_limits<std::size_t>::max();
    std::size_t marker = std::numeric_limits<std::size_t>::max() - 1UZ;
    if (PyObjectGuard lineStr(PyObject_GetAttrString(exception.get(), "lineno")); lineStr) {
        marker = PyLong_AsSize_t(lineStr);
        min    = marker > 5UZ ? marker - 5UZ : 0;
        max    = marker < (std::numeric_limits<std::size_t>::max() - 5UZ) ? marker + 5UZ : marker < std::numeric_limits<std::size_t>::max();
    }

    throw gr::exception(std::format("{}\nPython error: {}\n{}", msg, toString(exception), toLineCountAnnotated(pythonCode, min, max, marker)), location);
}

[[nodiscard]] inline std::string getDictionary(std::string_view moduleName) {
    PyObject* module = PyDict_GetItemString(PyImport_GetModuleDict(), moduleName.data());
    if (module == nullptr) {
        return "";
    }

    if (PyObject* module_dict = PyModule_GetDict(module); module_dict != nullptr) {
        PyObjectGuard dictGuard(PyObject_Repr(module_dict));
        return PyUnicode_AsUTF8(dictGuard);
    }
    return "";
}

template<typename T>
concept NoParamNoReturn = requires(T t) {
    { t() } -> std::same_as<void>;
};

template<typename T>
int numpyType() noexcept {
    // clang-format off
    if constexpr (std::is_same_v<T, bool>)          return NPY_BOOL;
    else if constexpr (std::is_same_v<T, std::int8_t>)   return NPY_BYTE;
    else if constexpr (std::is_same_v<T, std::uint8_t>)  return NPY_UBYTE;
    else if constexpr (std::is_same_v<T, std::int16_t>)  return NPY_SHORT;
    else if constexpr (std::is_same_v<T, std::uint16_t>) return NPY_USHORT;
    else if constexpr (std::is_same_v<T, std::int32_t>)  return NPY_INT;
    else if constexpr (std::is_same_v<T, std::uint32_t>) return NPY_UINT;
    else if constexpr (std::is_same_v<T, std::int64_t>)  return NPY_LONG;
    else if constexpr (std::is_same_v<T, std::uint64_t>) return NPY_ULONG;
    else if constexpr (std::is_same_v<T, float>)    return NPY_FLOAT;
    else if constexpr (std::is_same_v<T, double>)   return NPY_DOUBLE;
    else if constexpr (std::is_same_v<T, std::complex<float>>)  return NPY_CFLOAT;
    else if constexpr (std::is_same_v<T, std::complex<double>>) return NPY_CDOUBLE;
    else if constexpr (std::is_same_v<T, char*> || std::is_same_v<T, const char*>) return NPY_STRING;
    else return NPY_NOTYPE;
    // clang-format on
}

template<typename T>
requires std::is_arithmetic_v<T> || std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>>
constexpr inline PyObject* toPyArray(T* arrayData, std::initializer_list<std::size_t> dimensions) {
    assert(dimensions.size() >= 1 && "nDim needs to be >= 1");

    std::vector<npy_intp> npyDims(dimensions.begin(), dimensions.end());
    // N.B. reinterpret cast is needed to access NumPy's unsafe C-API
    void*     data    = const_cast<void*>(reinterpret_cast<const void*>(arrayData));
    PyObject* npArray = PyArray_SimpleNewFromData(static_cast<int>(dimensions.size()), npyDims.data(), python::numpyType<std::remove_cv_t<T>>(), data);
    if (!npArray) {
        python::throwCurrentPythonError("Unable to create NumPy array");
    }
    PyArray_CLEARFLAGS(reinterpret_cast<PyArrayObject*>(npArray), NPY_ARRAY_OWNDATA);

    if constexpr (!std::is_const_v<T>) {
        PyArray_ENABLEFLAGS(reinterpret_cast<PyArrayObject*>(npArray), NPY_ARRAY_WRITEABLE);
    } else {
        PyArray_CLEARFLAGS(reinterpret_cast<PyArrayObject*>(npArray), NPY_ARRAY_WRITEABLE);
    }
    return npArray;
}

template<typename T>
std::string sanitizedPythonBlockName() {
    std::string str = gr::meta::type_name<T>();
    std::replace(str.begin(), str.end(), ':', '_');
    std::replace(str.begin(), str.end(), '<', '_');
    std::replace(str.begin(), str.end(), '>', '_');
    str.erase(std::remove_if(str.begin(), str.end(), [](unsigned char c) { return std::isalnum(static_cast<unsigned char>(c)) == 0 && c != '_'; }), str.end());
    return str;
}

} // namespace gr::python
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

#include <fmt/format.h>
#include <stdexcept>
#include <vector>

namespace gr::python {

enum class EnforceFunction { MANDATORY, OPTIONAL };

class Interpreter {
    static std::atomic<std::size_t> _nInterpreters;
    static std::atomic<std::size_t> _nNumPyInit;
    static PyThreadState*           _interpreterThreadState;
    PyModuleDef*                    _moduleDefinitions;
    PyObject*                       _pMainModule; // borrowed reference
    PyObject*                       _pMainDict;   // borrowed reference
    PyObjectGuard                   _pCapsule;

public:
    template<typename T>
    explicit(false) Interpreter(T* classReference, PyModuleDef* moduleDefinitions = nullptr, std::source_location location = std::source_location::current()) : _moduleDefinitions(moduleDefinitions) {
        if (_nInterpreters.fetch_add(1UZ, std::memory_order_relaxed) == 0UZ) {
            Py_Initialize();
            if (PyErr_Occurred()) {
                PyErr_Print();
            }

            python::PyGILGuard guard;
            _interpreterThreadState = PyThreadState_Get();
            assert(_interpreterThreadState && "internal thread state is a nullptr");
            if (_nNumPyInit.fetch_add(1UZ, std::memory_order_relaxed) == 0UZ && _import_array() < 0) {
                // NumPy keeps internal state and does not allow to be re-initialised after 'Py_Finalize()' has been called.

                // initialise NumPy -- N.B. NumPy does not support sub-interpreters (as of Python 3.12):
                // "sys:1: UserWarning: NumPy was imported from a Python sub-interpreter but NumPy does not properly support sub-interpreters.
                // This will likely work for most users but might cause hard to track down issues or subtle bugs.
                // A common user of the rare sub-interpreter feature is wsgi which also allows single-interpreter mode.
                // Improvements in the case of bugs are welcome, but is not on the NumPy roadmap, and full support may require significant effort to achieve."
                python::throwCurrentPythonError("failed to initialize NumPy", location);
            }
        }
        assert(Py_IsInitialized() && "Python isn't properly initialised");
        // Ensure the Python GIL is initialized for this instance
        python::PyGILGuard localGuard;

        // need to be executed after the Python environment has been initialised
        _pMainModule = PyImport_AddModule("__main__");
        _pMainDict   = PyModule_GetDict(_pMainModule);
        if (classReference == nullptr || moduleDefinitions == nullptr) {
            return;
        }
        _pCapsule = PyObjectGuard(PyCapsule_New(static_cast<void*>(classReference), _moduleDefinitions->m_name, nullptr));
        if (!_pCapsule) {
            python::throwCurrentPythonError(fmt::format("Interpreter(*{}) - failed to create a capsule", gr::meta::type_name<T>()));
        }
        PyDict_SetItemString(_pMainDict, "capsule", _pCapsule);
        python::PyIncRef(_pCapsule); // need to explicitly increas count for the Python interpreter not to delete the reference by 'accident'

        // replaces the 'PyImport_AppendInittab("ClassName", &classDefinition)' to allow for other blocks being added
        // after the global Python interpreter is already being initialised
        PyObject* m = PyModule_Create(moduleDefinitions);
        if (m) {
            int ret = PyDict_SetItemString(PyImport_GetModuleDict(), moduleDefinitions->m_name, m);
            python::PyDecRef(m); // The module dict holds a reference now.
            if (ret != 0) {
                python::throwCurrentPythonError(fmt::format("Error inserting module {}.", _moduleDefinitions->m_name), location);
            }
        } else {
            python::throwCurrentPythonError(fmt::format("failed to create the module {}.", _moduleDefinitions->m_name), location);
        }
        if (PyDict_GetItemString(PyImport_GetModuleDict(), moduleDefinitions->m_name)) { // module successfully inserted - performing some additional checks
            assert(python::getDictionary(moduleDefinitions->m_name).size() > 0 && "dictionary exist for module");

            if (PyObject* imported_module = PyImport_ImportModule(moduleDefinitions->m_name); imported_module != nullptr) {
                python::PyDecRef(imported_module);
            } else {
                python::throwCurrentPythonError(fmt::format("Check import of {} failed.", _moduleDefinitions->m_name), location);
            }
        } else {
            python::throwCurrentPythonError(fmt::format("Manual import of {} failed.", _moduleDefinitions->m_name), location);
        }
    }

    ~Interpreter() {
        if (_nInterpreters.fetch_sub(1UZ, std::memory_order_acq_rel) == 1UZ && Py_IsInitialized()) {
            Py_Finalize();
        }
    }

    // Prevent copying and moving
    Interpreter(const Interpreter&)            = delete;
    Interpreter& operator=(const Interpreter&) = delete;
    Interpreter(Interpreter&&)                 = delete;
    Interpreter& operator=(Interpreter&&)      = delete;

    PyObject* getModule() { return _pMainModule; }

    PyObject* getDictionary() { return _pMainDict; }

    template<NoParamNoReturn Func>
    void invoke(Func func, std::string_view pythonCode = "", std::source_location location = std::source_location::current()) {
        assert(Py_IsInitialized());
        PyGILGuard localGuard;
        if (_interpreterThreadState != PyThreadState_Get()) {
            python::throwCurrentPythonError("detected sub-interpreter change which is not supported by NumPy", location, pythonCode);
        }
        if (PyErr_Occurred()) {
            python::throwCurrentPythonError("python::Interpreter::invoke() -- uncleared Python error before executing func", location, pythonCode);
        }

        func();

        if (PyErr_Occurred()) {
            python::throwCurrentPythonError("python::Interpreter::invoke() -- uncleared Python error after executing func", location, pythonCode);
        }
    }

    template<EnforceFunction forced = EnforceFunction::MANDATORY>
    python::PyObjectGuard invokeFunction(std::string_view functionName, PyObject* functionArguments = nullptr, std::source_location location = std::source_location::current()) {
        PyGILGuard localGuard;
        const bool hasFunction = PyObject_HasAttrString(getModule(), functionName.data());
        if constexpr (forced == EnforceFunction::MANDATORY) {
            if (!hasFunction) {
                python::throwCurrentPythonError(fmt::format("getFunction('{}', '{}') Python function not found or is not callable", functionName, python::toString(functionArguments)), location);
            }
        } else {
            if (!hasFunction) {
                return python::PyObjectGuard(nullptr);
            }
        }
        python::PyObjectGuard pyFunc(PyObject_GetAttrString(getModule(), functionName.data()));
        return python::PyObjectGuard(PyObject_CallObject(pyFunc, functionArguments));
    }
};

std::atomic<std::size_t> Interpreter::_nInterpreters{0UZ};
std::atomic<std::size_t> Interpreter::_nNumPyInit{0UZ};
PyThreadState*           Interpreter::_interpreterThreadState = nullptr;

} // namespace gr::python

#endif // GNURADIO_PYTHONINTERPRETER_HPP
