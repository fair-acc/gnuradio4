deps = [gr_core_dep, gr_basic_dep, fmt_dep, ut_dep]

qa_data_sink_exe = executable('qa_data_sink', 'qa_data_sink.cpp', dependencies : deps, install: false)
test('qa_data_sink', qa_data_sink_exe)

qa_selector_exe = executable('qa_selector', 'qa_selector.cpp', dependencies : deps, install: false)
test('qa_selector', qa_selector_exe)

qa_sources_exe = executable('qa_sources', 'qa_sources.cpp', dependencies : deps, install: false)
test('qa_sources', qa_sources_exe)
