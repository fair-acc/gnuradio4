
#include <gnuradio-4.0/basic/PythonBlock.hpp>

#include <boost/ut.hpp>

#include <gnuradio-4.0/Graph.hpp>

#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/meta/UnitTestHelper.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>

const boost::ut::suite<"python::<C-API abstraction interfaces>"> pythonInterfaceTests = [] {
    using namespace boost::ut;
    using namespace gr::python;

    "numpyType<T>()"_test = [] {
        expect(numpyType<bool>() == NPY_BOOL);
        expect(numpyType<int8_t>() == NPY_BYTE);
        expect(numpyType<uint8_t>() == NPY_UBYTE);
        expect(numpyType<int16_t>() == NPY_SHORT);
        expect(numpyType<uint16_t>() == NPY_USHORT);
        expect(numpyType<int32_t>() == NPY_INT);
        expect(numpyType<uint32_t>() == NPY_UINT);
        expect(numpyType<int64_t>() == NPY_LONG);
        expect(numpyType<uint64_t>() == NPY_ULONG);
        expect(numpyType<float>() == NPY_FLOAT);
        expect(numpyType<double>() == NPY_DOUBLE);
        expect(numpyType<std::complex<float>>() == NPY_CFLOAT);
        expect(numpyType<std::complex<double>>() == NPY_CDOUBLE);
        expect(numpyType<char*>() == NPY_STRING);
        expect(numpyType<const char*>() == NPY_STRING);
        expect(numpyType<void>() == NPY_NOTYPE);
    };

    // N.B. every case keeps one extra reference of its own, so the refcount can still be read after the guards die.
    auto makeUniqueObject = [] { return PyLong_FromLong(987654321L); }; // outside CPython's small-int cache, so never shared

    "moving a guard transfers sole ownership"_test = [&] {
        Interpreter interpreter{static_cast<int*>(nullptr)};
        PyGILGuard  gil;

        PyObjectGuard keepAlive(makeUniqueObject());
        PyIncRef(keepAlive.get()); // the reference the moved guard will own
        {
            PyObjectGuard source(keepAlive.get());
            PyObjectGuard target(std::move(source));
            expect(target.get() == keepAlive.get());
            expect(source.get() == nullptr) << "a moved-from guard must not retain the pointer";
            expect(eq(PyRefCount(keepAlive.get()), Py_ssize_t{2})) << "moving must neither add nor drop a reference";
        }
        expect(eq(PyRefCount(keepAlive.get()), Py_ssize_t{1})) << "the moved reference must be released exactly once";
    };

    "move-assigning a guard releases the overwritten reference"_test = [&] {
        Interpreter interpreter{static_cast<int*>(nullptr)};
        PyGILGuard  gil;

        PyObjectGuard overwritten(makeUniqueObject());
        PyObjectGuard assigned(PyLong_FromLong(123456789L));
        PyIncRef(overwritten.get());
        PyIncRef(assigned.get());
        {
            PyObjectGuard target(overwritten.get());
            PyObjectGuard source(assigned.get());
            target = std::move(source);
            expect(eq(PyRefCount(overwritten.get()), Py_ssize_t{1})) << "the overwritten reference must be released once, not swapped into the source";
            expect(eq(PyRefCount(assigned.get()), Py_ssize_t{2}));
            expect(source.get() == nullptr) << "a moved-from guard must not retain the pointer";
        }
        expect(eq(PyRefCount(assigned.get()), Py_ssize_t{1}));
        expect(eq(PyRefCount(overwritten.get()), Py_ssize_t{1}));
    };

    "copy-assigning a guard releases the overwritten reference"_test = [&] {
        Interpreter interpreter{static_cast<int*>(nullptr)};
        PyGILGuard  gil;

        PyObjectGuard overwritten(makeUniqueObject());
        PyObjectGuard shared(PyLong_FromLong(123456789L));
        PyIncRef(overwritten.get());
        {
            PyObjectGuard target(overwritten.get());
            PyIncRef(shared.get()); // the reference 'source' owns
            PyObjectGuard source(shared.get());
            target = source;
            expect(eq(PyRefCount(overwritten.get()), Py_ssize_t{1})) << "the overwritten reference must not leak";
            expect(eq(PyRefCount(shared.get()), Py_ssize_t{3})) << "copying must add exactly one reference";
        }
        expect(eq(PyRefCount(shared.get()), Py_ssize_t{1}));
    };

    "self-assigning a guard keeps its reference alive"_test = [&] {
        Interpreter interpreter{static_cast<int*>(nullptr)};
        PyGILGuard  gil;

        PyObjectGuard keepAlive(makeUniqueObject());
        PyIncRef(keepAlive.get());
        {
            PyObjectGuard  guard(keepAlive.get());
            PyObjectGuard& alias = guard; // via a reference, so the self-assignment is not diagnosed at compile time
            guard                = alias;
            expect(eq(PyRefCount(keepAlive.get()), Py_ssize_t{2})) << "self-assignment must not change the count";
            expect(guard.get() == keepAlive.get()) << "self-assignment must not release the object";
        }
        expect(eq(PyRefCount(keepAlive.get()), Py_ssize_t{1}));
    };
};

const boost::ut::suite<"PythonBlock"> pythonBlockTests = [] {
    using namespace boost::ut;
    using namespace gr::basic;
    using namespace std::string_literals;
    using namespace std::string_view_literals;

    static_assert(gr::HasRequiredProcessFunction<gr::basic::PythonBlock<std::int32_t>>);
    static_assert(gr::HasProcessBulkFunction<gr::basic::PythonBlock<std::int32_t>>);
    static_assert(gr::HasRequiredProcessFunction<gr::basic::PythonBlock<float>>);
    static_assert(gr::HasProcessBulkFunction<gr::basic::PythonBlock<float>>);

    "nominal PoC"_test = [] {
        // Your Python script
        std::string python_script = R"(import time;
counter = 0

def process_bulk(ins, outs):
    global counter
    start = time.time()
    print('Start Python processing iteration: {}'.format(counter))
    # Print current settings
    settings = this_block.getSettings()
    print("Current settings:", settings)

    # tag handling
    if this_block.tagAvailable():
        tag = this_block.getTag()
        print('Tag:', tag)

    counter += 1
    # process the input->output samples
    for i in range(len(ins)):
        outs[i][:] = ins[i] * 2

    # Update settings with the counter
    settings["counter"] = str(counter)
    this_block.setSettings(settings)

    print('Stop Python processing - time: {} seconds'.format(time.time() - start))
)";

        PythonBlock<std::int32_t> myBlock({{"n_inputs", 2U}, {"n_outputs", 2U}, {"python_script", python_script}});
        myBlock.init(myBlock.progress); // needed for unit-test only when executed outside a Scheduler/Graph

        int                                        count = 0;
        std::vector<std::int32_t>                  data1 = {1, 2, 3};
        std::vector<std::int32_t>                  data2 = {4, 5, 6};
        std::vector<std::int32_t>                  out1(3);
        std::vector<std::int32_t>                  out2(3);
        std::vector<std::span<std::int32_t>>       outs    = {out1, out2};
        std::vector<std::span<const std::int32_t>> ins     = {data1, data2};
        std::span<std::span<const std::int32_t>>   spanIns = ins;
        for (const auto& span : ins) {
            std::println("InPort[{}] : [{}]", count++, gr::join(span, ", "));
        }
        std::println("");

        for (std::size_t i = 0; i < 3; i++) {
            std::println("C++ processing iteration: {}", i);
            std::vector<std::span<const std::int32_t>> constOuts(outs.begin(), outs.end());
            std::span<std::span<const std::int32_t>>   constSpanOuts = constOuts;
            std::span<std::span<std::int32_t>>         spanOuts      = outs;

            try {
                if (i == 0) {
                    myBlock.processBulk(spanIns, spanOuts);
                } else {
                    myBlock.processBulk(constSpanOuts, spanOuts);
                }
            } catch (const std::exception& ex) {
                std::println(stderr, "myBlock.processBulk(...) - threw unexpected exception:\n {}", ex.what());
                expect(false) << "nominal example should not throw";
            }

            std::println("C++ side got:");
            std::println("settings: {}", myBlock._settingsMap);
            for (const auto& span : outs) {
                std::println("OutPort[{}] : [{}]", count++, gr::join(span, ", "));
            }
            std::println("");
        }

        expect(eq(outs[0][0], 8)) << "out1[0] should be 8";
        expect(eq(outs[0][1], 16)) << "out1[1] should be 16";
        expect(eq(outs[0][2], 24)) << "out1[2] should be 24";

        expect(eq(outs[1][0], 32)) << "out2[0] should be 32";
        expect(eq(outs[1][1], 40)) << "out2[1] should be 40";
        expect(eq(outs[1][2], 48)) << "out2[2] should be 48";

        expect(eq(myBlock.getSettings().at("counter"), "3"s));
    };

    "Python SyntaxError"_test = [] {
        // Your Python script
        std::string python_script = R"(def process_bulk(ins, outs):

    # process the input->output samples
    for i in range(len(ins))     # <- (N.B. missing ':')
        outs[i][:] = ins[i] * 2
)";

        PythonBlock<std::int32_t> myBlock({{"n_inputs", 2U}, {"n_outputs", 2U}, {"python_script", python_script}});

        bool throws = false;
        try {
            myBlock.settings().init();
            std::ignore = myBlock.settings().applyStagedParameters(); // needed for unit-test only when executed outside a Scheduler/Graph
        } catch (const std::exception& ex) {
            throws = true;
            std::println("myBlock.processBulk(...) - correctly threw SyntaxError exception:\n {}", ex.what());
        }
        expect(throws) << "SyntaxError should throw";
    };

    "Python RuntimeWarning as exception"_test = [] {
        // Your Python script
        std::string python_script = R"(def process_bulk(ins, outs):

    # process the input->output samples
    for i in range(len(ins)):
        outs[i][:] = ins[i] * 2/0 # <- (N.B. division by zero)
)";

        PythonBlock<float> myBlock({{"n_inputs", 2U}, {"n_outputs", 2U}, {"python_script", python_script}});
        myBlock.init(myBlock.progress); // needed for unit-test only when executed outside a Scheduler/Graph

        std::vector<float>                  data1 = {1, 2, 3};
        std::vector<float>                  data2 = {4, 5, 6};
        std::vector<float>                  out1(3);
        std::vector<float>                  out2(3);
        std::vector<std::span<float>>       outs = {out1, out2};
        std::vector<std::span<const float>> ins  = {data1, data2};

        bool throws = false;
        try {
            myBlock.processBulk(std::span(ins), std::span(outs));
        } catch (const std::exception& ex) {
            throws = true;
            std::println("myBlock.processBulk(...) - correctly threw RuntimeWarning as exception:\n {}", ex.what());
        }
        expect(throws) << "RuntimeWarning should throw";
    };

    "a block configured with no port counts defaults to one in and one out"_test = [] {
        std::string python_script = "def process_bulk(ins, outs):\n    for i in range(len(ins)):\n        outs[i][:] = ins[i] * 3\n";

        PythonBlock<std::int32_t> myBlock({{"python_script", python_script}}); // no n_inputs / n_outputs
        myBlock.init(myBlock.progress);

        expect(eq(myBlock.n_inputs, gr::Size_t{1})) << "n_inputs must default to its lower limit, not 0";
        expect(eq(myBlock.n_outputs, gr::Size_t{1})) << "n_outputs must default to its lower limit, not 0";
        expect(eq(myBlock.inputs.size(), 1UZ)) << "the default port count must actually be applied";
        expect(eq(myBlock.outputs.size(), 1UZ)) << "the default port count must actually be applied";

        std::vector<std::int32_t>                  data = {1, 2, 3};
        std::vector<std::int32_t>                  out(3);
        std::vector<std::span<const std::int32_t>> ins  = {data};
        std::vector<std::span<std::int32_t>>       outs = {out};
        myBlock.processBulk(std::span(ins), std::span(outs));
        expect(eq(out, std::vector<std::int32_t>{3, 6, 9})) << std::format("default-configured block produced {}", out);
    };

    "a script without process_bulk is rejected"_test = [] {
        std::string python_script = "def some_other_name(ins, outs):\n    pass\n";

        PythonBlock<std::int32_t> myBlock({{"n_inputs", 1U}, {"n_outputs", 1U}, {"python_script", python_script}});
        myBlock.init(myBlock.progress);

        std::vector<std::int32_t>                  data = {1, 2, 3};
        std::vector<std::int32_t>                  out(3);
        std::vector<std::span<const std::int32_t>> ins  = {data};
        std::vector<std::span<std::int32_t>>       outs = {out};

        bool throws = false;
        try { // N.B. settings are applied lazily, so the script is validated on the first processBulk rather than in init()
            myBlock.processBulk(std::span(ins), std::span(outs));
        } catch (const std::exception& ex) {
            throws = true;
            expect(std::string_view(ex.what()).contains("process_bulk")) << std::format("the error should name the missing function, got: {}", ex.what());
        }
        expect(throws) << "a script without process_bulk must be rejected";
    };

    "lifecycle callbacks reach the script"_test = [] {
        std::string               python_script = R"(def record(step):
    settings = this_block.getSettings()
    settings["lifecycle"] = settings.get("lifecycle", "") + step + ";"
    this_block.setSettings(settings)

def process_bulk(ins, outs):
    for i in range(len(ins)):
        outs[i][:] = ins[i]

def start():
    record("start")

def pause():
    record("pause")

def resume():
    record("resume")

def stop():
    record("stop")
)";
        PythonBlock<std::int32_t> myBlock({{"n_inputs", 1U}, {"n_outputs", 1U}, {"python_script", python_script}});
        myBlock.init(myBlock.progress);

        myBlock.start();
        myBlock.pause();
        myBlock.resume();
        myBlock.stop();

        expect(myBlock.getSettings().contains("lifecycle")) << "no lifecycle callback reached the script";
        expect(eq(myBlock.getSettings().at("lifecycle"), "start;pause;resume;stop;"s)) << "lifecycle callbacks ran in the wrong order or not at all";
    };

    "an absent lifecycle callback is optional, not an error"_test = [] {
        std::string python_script = "def process_bulk(ins, outs):\n    for i in range(len(ins)):\n        outs[i][:] = ins[i]\n"; // defines no start/stop/...

        PythonBlock<std::int32_t> myBlock({{"n_inputs", 1U}, {"n_outputs", 1U}, {"python_script", python_script}});
        myBlock.init(myBlock.progress);

        bool throws = false;
        try {
            myBlock.start();
            myBlock.pause();
            myBlock.resume();
            myBlock.stop();
            myBlock.reset();
        } catch (const std::exception& ex) {
            throws = true;
            std::println("unexpected: {}", ex.what());
        }
        expect(!throws) << "lifecycle hooks are optional and must not throw when the script omits them";
    };

    "two blocks with different scripts keep separate namespaces"_test = [] {
        // both scripts define 'process_bulk' and bind 'this_block'; sharing one namespace made the last one configured win
        std::string doubleIt = "def process_bulk(ins, outs):\n    for i in range(len(ins)):\n        outs[i][:] = ins[i] * 2\n";
        std::string tenTimes = "def process_bulk(ins, outs):\n    for i in range(len(ins)):\n        outs[i][:] = ins[i] * 10\n";

        PythonBlock<std::int32_t> doublingBlock({{"n_inputs", 1U}, {"n_outputs", 1U}, {"python_script", doubleIt}});
        doublingBlock.init(doublingBlock.progress);
        PythonBlock<std::int32_t> scalingBlock({{"n_inputs", 1U}, {"n_outputs", 1U}, {"python_script", tenTimes}});
        scalingBlock.init(scalingBlock.progress);

        std::vector<std::int32_t>                  data = {1, 2, 3};
        std::vector<std::int32_t>                  doubled(3);
        std::vector<std::int32_t>                  scaled(3);
        std::vector<std::span<const std::int32_t>> doublingIn  = {data};
        std::vector<std::span<const std::int32_t>> scalingIn   = {data};
        std::vector<std::span<std::int32_t>>       doublingOut = {doubled};
        std::vector<std::span<std::int32_t>>       scalingOut  = {scaled};

        doublingBlock.processBulk(std::span(doublingIn), std::span(doublingOut));
        scalingBlock.processBulk(std::span(scalingIn), std::span(scalingOut));

        expect(eq(doubled, std::vector<std::int32_t>{2, 4, 6})) << std::format("the doubling block ran the wrong script: {}", doubled);
        expect(eq(scaled, std::vector<std::int32_t>{10, 20, 30})) << std::format("the scaling block ran the wrong script: {}", scaled);
    };

    "non-string setting values raise TypeError instead of crashing"_test = [] {
        std::string               python_script = R"(def process_bulk(ins, outs):
    this_block.setSettings({"answer": 42})  # <- int value, not a string
)";
        PythonBlock<std::int32_t> myBlock({{"n_inputs", 1U}, {"n_outputs", 1U}, {"python_script", python_script}});
        myBlock.init(myBlock.progress);

        std::vector<std::int32_t>                  data = {1, 2, 3};
        std::vector<std::int32_t>                  out(3);
        std::vector<std::span<const std::int32_t>> ins  = {data};
        std::vector<std::span<std::int32_t>>       outs = {out};

        bool throws = false;
        try {
            myBlock.processBulk(std::span(ins), std::span(outs));
        } catch (const std::exception& ex) {
            throws = true;
            expect(std::string_view(ex.what()).contains("TypeError")) << std::format("expected a Python TypeError, got: {}", ex.what());
        }
        expect(throws) << "a non-string settings value must surface as a Python error";
    };

    "a foreign capsule is rejected instead of dereferenced"_test = [] {
        std::string               python_script = R"(import ctypes
def process_bulk(ins, outs):
    this_block.getTag.__self__.capsule = ctypes.pythonapi.PyCapsule_New(ctypes.c_void_p(1), b"bogus", None)
    this_block.getTag()  # <- capsule name no longer matches this block type
)";
        PythonBlock<std::int32_t> myBlock({{"n_inputs", 1U}, {"n_outputs", 1U}, {"python_script", python_script}});
        myBlock.init(myBlock.progress);

        std::vector<std::int32_t>                  data = {1, 2, 3};
        std::vector<std::int32_t>                  out(3);
        std::vector<std::span<const std::int32_t>> ins  = {data};
        std::vector<std::span<std::int32_t>>       outs = {out};

        bool throws = false;
        try {
            myBlock.processBulk(std::span(ins), std::span(outs));
        } catch (const std::exception&) {
            throws = true; // a Python-level error is the correct outcome; a crash or C++ unwind through CPython is not
        }
        expect(throws) << "a capsule of the wrong type must produce a Python error";
    };

    "Python Execution via Scheduler/Graph"_test = [] {
        std::string python_script = R"(def process_bulk(ins, outs):

    # process the input->output samples
    for i in range(len(ins)):
        outs[i][:] = ins[i] * 2
)";

        using namespace gr::testing;
        Graph graph;
        auto& src   = graph.emplaceBlock<TagSource<int32_t>>({{"n_samples_max", 5U}, {"mark_tag", false}});
        auto& block = graph.emplaceBlock<PythonBlock<int32_t>>({{"n_inputs", 1U}, {"n_outputs", 1U}, {"python_script", python_script}});
        auto& sink  = graph.emplaceBlock<TagSink<int32_t, ProcessFunction::USE_PROCESS_BULK>>({{"n_samples_expected", 5U}, {"verbose_console", true}});

        expect(graph.connect(src, "out", block, "inputs#0").has_value());
        expect(graph.connect(block, "outputs#0", sink, "in").has_value());

        gr::scheduler::Simple sched;
        if (auto ret = sched.exchange(std::move(graph)); !ret) {
            throw std::runtime_error(std::format("failed to initialize scheduler: {}", ret.error()));
        }

        bool throws = false;
        try {
            expect(sched.runAndWait().has_value());
        } catch (const std::exception& ex) {
            throws = true;
            std::println("sched.runAndWait() unexpectedly threw an exception:\n {}", ex.what());
        }
        expect(!throws);

        expect(eq(sink._nSamplesProduced, 5U)) << "sinkOne did not consume enough input samples";
        expect(eq(sink._samples, std::vector<std::int32_t>{0, 2, 4, 6, 8})) << std::format("mismatch of vector {}", sink._samples);
    };

    "Python Execution - Lifecycle method tests"_test = [] {
        std::string python_script = R"x(import os
counter = 0

# optional life-cycle methods - can be used to inform the block of the scheduling state
def start():
    global counter
    print("Python: invoked start")
    counter += 1

def stop():
    global counter
    print("Python: invoked stop")
    counter += 1

def pause():
    global counter
    counter += 1

def resume():
    global counter
    counter += 1

def reset():
    global counter
    counter += 1

# stream-based processing
def process_bulk(ins, outs):
    global counter
    assert counter == 4, "Counter is not equal to 4 (N.B. having called start(), pause(), resume(), reset() callback functions"

    print("Python: invoked process_bulk(..)")
    # process the input->output samples
    for i in range(len(ins)):
        outs[i][:] = ins[i] * 2
)x";

        using namespace gr::testing;
        Graph graph;
        auto& src   = graph.emplaceBlock<TagSource<float>>({{"n_samples_max", 5U}, {"mark_tag", false}});
        auto& block = graph.emplaceBlock<PythonBlock<float>>({{"n_inputs", 1U}, {"n_outputs", 1U}, {"python_script", python_script}});
        auto& sink  = graph.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_BULK>>({{"n_samples_expected", 5U}, {"verbose_console", true}});

        expect(graph.connect(src, "out", block, "inputs#0").has_value());
        expect(graph.connect(block, "outputs#0", sink, "in").has_value());

        gr::scheduler::Simple sched;
        if (auto ret = sched.exchange(std::move(graph)); !ret) {
            throw std::runtime_error(std::format("failed to initialize scheduler: {}", ret.error()));
        }

        block.pause();  // simplified calling
        block.resume(); // simplified calling
        block.reset();  // simplified calling
        bool throws = false;
        try {
            expect(sched.runAndWait().has_value());
        } catch (const std::exception& ex) {
            throws = true;
            std::println("sched.runAndWait() unexpectedly threw an exception:\n {}", ex.what());
        }
        expect(!throws);

        expect(eq(sink._nSamplesProduced, 5U)) << "sinkOne did not consume enough input samples";
        expect(eq(sink._samples, std::vector<float>{0.f, 2.f, 4.f, 6.f, 8.f})) << std::format("mismatch of vector {}", sink._samples);
    };
};

int main() { /* tests are statically executed */ }
