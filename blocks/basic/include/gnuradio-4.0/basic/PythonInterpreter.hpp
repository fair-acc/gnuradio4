#ifndef GNURADIO_PYTHONINTERPRETER_HPP
#define GNURADIO_PYTHONINTERPRETER_HPP

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
#ifndef __clang__
#pragma GCC diagnostic ignored "-Wuseless-cast"
#endif
#endif
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <atomic>
#include <cassert>
#include <cctype>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <mutex>
#include <regex>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>

#include <gnuradio-4.0/Message.hpp>

#include <numpy/numpyconfig.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

namespace gr::python {

inline static PyObject* TrueObj  = Py_True;
inline static PyObject* FalseObj = Py_False;
inline static PyObject* NoneObj  = Py_None;

inline bool isPyDict(const PyObject* obj) { return PyDict_Check(obj); }

inline void PyDecRef(PyObject* obj) { // wrapper to isolate unsafe warning on C-API casts
    Py_XDECREF(obj);
}

inline void PyIncRef(PyObject* obj) { // wrapper to isolate unsafe warning on C-API casts
    Py_XINCREF(obj);
}

inline std::string PyBytesAsString(PyObject* op) {
    const char* bytes = PyBytes_AsString(op);
    return bytes == nullptr ? std::string{} : std::string{bytes};
}

// runs a CPython callback body; a C++ exception must never unwind through CPython's C frames, so it becomes a Python error instead
template<typename TFunc>
PyObject* invokeCallback(TFunc&& body) noexcept {
    try {
        return body();
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    } catch (...) {
        PyErr_SetString(PyExc_RuntimeError, "unknown C++ exception raised in a PythonBlock callback");
        return nullptr;
    }
}

inline Py_ssize_t PyRefCount(PyObject* obj) { // wrapper to isolate unsafe warning on C-API casts, cf. PyIncRef/PyDecRef
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
#ifndef __clang__
#pragma GCC diagnostic ignored "-Wuseless-cast"
#endif
#endif
    return Py_REFCNT(obj);
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
}

// owns one strong reference to a PyObject. A moved-from guard owns nothing, so exactly one release happens per reference.
class PyObjectGuard {
    PyObject* _ptr;

public:
    explicit PyObjectGuard(PyObject* ptr = nullptr) : _ptr(ptr) {}

    PyObjectGuard(const PyObjectGuard& other) : _ptr(other._ptr) { PyIncRef(_ptr); }

    PyObjectGuard(PyObjectGuard&& other) noexcept : _ptr(std::exchange(other._ptr, nullptr)) {}

    ~PyObjectGuard() { PyDecRef(_ptr); }

    PyObjectGuard& operator=(const PyObjectGuard& other) {
        if (this != &other) {
            PyIncRef(other._ptr); // increment before releasing, so self-referential aliases stay alive
            PyDecRef(_ptr);
            _ptr = other._ptr;
        }
        return *this;
    }

    PyObjectGuard& operator=(PyObjectGuard&& other) noexcept {
        if (this != &other) {
            PyDecRef(_ptr);
            _ptr = std::exchange(other._ptr, nullptr);
        }
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
    if (object == nullptr) {
        return "<nullptr>";
    }
    PyObjectGuard strObj(PyObject_Repr(object));
    if (!strObj) {
        PyErr_Clear();
        return "<object without a repr>";
    }
    PyObjectGuard bytesObj(PyUnicode_AsEncodedString(strObj.get(), "utf-8", "strict"));
    if (!bytesObj) {
        PyErr_Clear();
        return "<repr that is not valid UTF-8>";
    }
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
    for (std::size_t i = min; i < std::min(lines.size(), max); i++) {
        // N.B. Python counts lines from '1', so report i + 1 and compare the 1-based marker against it
        annotatedCode += std::format("{:3}:{}{}\n", i + 1UZ, lines[i], i + 1UZ == marker ? "   ####### <== here's your problem #######" : "");
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
    // std::println("detailed debug info: {}", getDebugPythonObjectAttributes(exception))

    std::size_t min    = 0UZ;
    std::size_t max    = std::numeric_limits<std::size_t>::max();
    std::size_t marker = std::numeric_limits<std::size_t>::max() - 1UZ;
    if (PyObjectGuard lineStr(PyObject_GetAttrString(exception.get(), "lineno")); lineStr) {
        marker = PyLong_AsSize_t(lineStr);
        if (PyErr_Occurred()) { // 'lineno' need not be an integer
            PyErr_Clear();
            marker = std::numeric_limits<std::size_t>::max() - 1UZ;
        } else {
            min = marker > 5UZ ? marker - 5UZ : 0UZ;
            max = marker < (std::numeric_limits<std::size_t>::max() - 5UZ) ? marker + 5UZ : std::numeric_limits<std::size_t>::max();
        }
    } else {
        PyErr_Clear(); // only syntax-like exceptions carry a 'lineno'; leaving the probe's AttributeError set would misreport the next call
    }

    throw gr::exception(std::format("{}\nPython error: {}\n{}", msg, toString(exception), toLineCountAnnotated(pythonCode, min, max, marker)), location);
}

[[nodiscard]] inline std::string getDictionary(std::string_view moduleName) {
    const std::string name(moduleName); // the C-API needs a NUL-terminated string, which string_view::data() does not promise
    PyObject*         module = PyDict_GetItemString(PyImport_GetModuleDict(), name.c_str());
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
inline PyObject* toPyArray(T* arrayData, std::initializer_list<std::size_t> dimensions) {
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
    std::string str(gr::meta::type_name<T>());
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

#include <format>
#include <stdexcept>
#include <vector>

namespace gr::python {

enum class EnforceFunction { MANDATORY, OPTIONAL };

// CPython and NumPy do not survive a 'Py_Finalize()'/'Py_Initialize()' cycle: NumPy's C-API state is bound to the interpreter that first
// imported it and is never re-imported, so a later interpreter frees objects belonging to the earlier one. The interpreter is therefore
// kept alive for the whole process and finalised exactly once, here.
inline void finalizeInterpreterAtExit();

class Interpreter {
    static std::once_flag           _initialiseOnce;
    static std::atomic<std::size_t> _nModulesCreated;
    static PyInterpreterState*      _interpreterState;
    static PyThreadState*           _mainThreadState;
    PyModuleDef*                    _moduleDefinitions;
    PyObjectGuard                   _blockModule;         // private per-block module: two blocks must not share one namespace
    PyObject*                       _pBlockDict{nullptr}; // borrowed from _blockModule
    PyObjectGuard                   _pCapsule;

public:
    template<typename T>
    explicit(false) Interpreter(T* classReference, PyModuleDef* moduleDefinitions = nullptr, std::source_location location = std::source_location::current()) : _moduleDefinitions(moduleDefinitions) {
        // once per process, and never again: NumPy keeps internal state and cannot be re-initialised after 'Py_Finalize()'.
        // N.B. NumPy does not support sub-interpreters (as of Python 3.12):
        // "sys:1: UserWarning: NumPy was imported from a Python sub-interpreter but NumPy does not properly support sub-interpreters.
        // This will likely work for most users but might cause hard to track down issues or subtle bugs.
        // A common user of the rare sub-interpreter feature is wsgi which also allows single-interpreter mode.
        // Improvements in the case of bugs are welcome, but is not on the NumPy roadmap, and full support may require significant effort to achieve."
        std::call_once(_initialiseOnce, [&location] {
            Py_Initialize();
            if (PyErr_Occurred()) {
                PyErr_Print();
            }
            if (_import_array() < 0) {
                python::throwCurrentPythonError("failed to initialize NumPy", location);
            }
            _interpreterState = PyInterpreterState_Get();
            assert(_interpreterState && "interpreter state is a nullptr");
            std::atexit(finalizeInterpreterAtExit);
            _mainThreadState = PyEval_SaveThread(); // Py_Initialize() leaves the GIL held; release it so any thread can PyGILState_Ensure()
        });
        assert(Py_IsInitialized() && "Python isn't properly initialised");
        // Ensure the Python GIL is initialized for this instance
        python::PyGILGuard localGuard;

        // N.B. a private module, not the shared '__main__': each block needs its own 'process_bulk' and 'this_block'.
        const std::string blockModuleName = std::format("gr_python_block_{}", _nModulesCreated.fetch_add(1UZ, std::memory_order_relaxed));
        _blockModule                      = PyObjectGuard(PyModule_New(blockModuleName.c_str()));
        if (!_blockModule) {
            python::throwCurrentPythonError(std::format("failed to create the private Python module '{}'", blockModuleName), location);
        }
        _pBlockDict = PyModule_GetDict(_blockModule.get());
        if (PyDict_SetItemString(_pBlockDict, "__builtins__", PyEval_GetBuiltins()) != 0) { // a module made by PyModule_New has no builtins yet
            python::throwCurrentPythonError(std::format("failed to provide __builtins__ to '{}'", blockModuleName), location);
        }
        if (classReference == nullptr || moduleDefinitions == nullptr) {
            return;
        }
        _pCapsule = PyObjectGuard(PyCapsule_New(static_cast<void*>(classReference), _moduleDefinitions->m_name, nullptr));
        if (!_pCapsule) {
            python::throwCurrentPythonError(std::format("Interpreter(*{}) - failed to create a capsule", gr::meta::type_name<T>()));
        }
        PyDict_SetItemString(_pBlockDict, "capsule", _pCapsule);
        python::PyIncRef(_pCapsule); // need to explicitly increas count for the Python interpreter not to delete the reference by 'accident'

        // replaces the 'PyImport_AppendInittab("ClassName", &classDefinition)' to allow for other blocks being added
        // after the global Python interpreter is already being initialised
        PyObject* m = PyModule_Create(moduleDefinitions);
        if (m) {
            int ret = PyDict_SetItemString(PyImport_GetModuleDict(), moduleDefinitions->m_name, m);
            python::PyDecRef(m); // The module dict holds a reference now.
            if (ret != 0) {
                python::throwCurrentPythonError(std::format("Error inserting module {}.", _moduleDefinitions->m_name), location);
            }
        } else {
            python::throwCurrentPythonError(std::format("failed to create the module {}.", _moduleDefinitions->m_name), location);
        }
        if (PyDict_GetItemString(PyImport_GetModuleDict(), moduleDefinitions->m_name)) { // module successfully inserted - performing some additional checks
            assert(python::getDictionary(moduleDefinitions->m_name).size() > 0 && "dictionary exist for module");

            if (PyObject* imported_module = PyImport_ImportModule(moduleDefinitions->m_name); imported_module != nullptr) {
                python::PyDecRef(imported_module);
            } else {
                python::throwCurrentPythonError(std::format("Check import of {} failed.", _moduleDefinitions->m_name), location);
            }
        } else {
            python::throwCurrentPythonError(std::format("Manual import of {} failed.", _moduleDefinitions->m_name), location);
        }
    }

    ~Interpreter() {
        if (!Py_IsInitialized()) {
            return;
        }
        PyGILGuard guard; // releasing a PyObject without the GIL corrupts its reference count
        _pCapsule    = PyObjectGuard{};
        _blockModule = PyObjectGuard{};
    }

    // Prevent copying and moving
    Interpreter(const Interpreter&)            = delete;
    Interpreter& operator=(const Interpreter&) = delete;
    Interpreter(Interpreter&&)                 = delete;
    Interpreter& operator=(Interpreter&&)      = delete;

    static PyThreadState* mainThreadState() noexcept { return _mainThreadState; }

    PyObject* getModule() { return _blockModule.get(); }

    PyObject* getDictionary() { return _pBlockDict; }

    template<NoParamNoReturn Func>
    void invoke(Func func, std::string_view pythonCode = "", std::source_location location = std::source_location::current()) {
        assert(Py_IsInitialized());
        PyGILGuard localGuard;
        if (PyInterpreterState_Get() != _interpreterState) { // the interpreter, not the thread: every thread has its own state
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
    [[nodiscard]] python::PyObjectGuard invokeFunction(std::string_view functionName, PyObject* functionArguments = nullptr, std::source_location location = std::source_location::current()) {
        if (getModule() == nullptr) { // ~Block() invokes stop() after the derived _interpreter member is gone
            return python::PyObjectGuard(nullptr);
        }
        PyGILGuard        localGuard;
        const std::string function(functionName); // the C-API needs a NUL-terminated string, which string_view::data() does not promise
        const bool        hasFunction = PyObject_HasAttrString(getModule(), function.c_str());
        if constexpr (forced == EnforceFunction::MANDATORY) {
            if (!hasFunction) {
                python::throwCurrentPythonError(std::format("getFunction('{}', '{}') Python function not found or is not callable", functionName, python::toString(functionArguments)), location);
            }
        } else {
            if (!hasFunction) {
                return python::PyObjectGuard(nullptr);
            }
        }
        python::PyObjectGuard pyFunc(PyObject_GetAttrString(getModule(), function.c_str()));
        return python::PyObjectGuard(PyObject_CallObject(pyFunc, functionArguments));
    }
};

inline std::once_flag           Interpreter::_initialiseOnce;
inline std::atomic<std::size_t> Interpreter::_nModulesCreated{0UZ};
inline PyInterpreterState*      Interpreter::_interpreterState = nullptr;
inline PyThreadState*           Interpreter::_mainThreadState  = nullptr;

inline void finalizeInterpreterAtExit() {
    if (Py_IsInitialized()) {
        PyEval_RestoreThread(Interpreter::mainThreadState()); // Py_Finalize() requires the GIL, which the constructor released
        Py_Finalize();
    }
}

} // namespace gr::python

#endif // GNURADIO_PYTHONINTERPRETER_HPP
