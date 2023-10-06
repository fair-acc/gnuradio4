deps = [gr_core_dep, fmt_dep, reflcpp_dep, ut_dep, gr_basic_dep, gr_testing_dep]

qa_buffer_exe = executable('qa_buffer', 'qa_buffer.cpp', dependencies : [ deps ], install: false)
test('qa_buffer', qa_buffer_exe)

qa_dynamic_node_exe = executable('qa_dynamic_node', 'qa_dynamic_node.cpp', dependencies : [ deps ], install: false)
test('qa_dynamic_node', qa_dynamic_node_exe)

qa_dynamic_port_exe = executable('qa_dynamic_port', 'qa_dynamic_port.cpp', dependencies : [ deps ], install: false)
test('qa_dynamic_port', qa_dynamic_port_exe)

qa_hier_node_exe = executable('qa_hier_node', 'qa_hier_node.cpp', dependencies : [ deps ], install: false)
test('qa_hier_node', qa_hier_node_exe)

qa_node_exe = executable('qa_node', 'qa_node.cpp', dependencies : [ deps ], install: false)
test('qa_node', qa_node_exe)

qa_plugins_test_exe = executable('qa_plugins_test', 'qa_plugins_test.cpp', dependencies : [ deps ], install: false)
test('qa_plugins_test', qa_plugins_test_exe)

qa_reader_writer_lock_exe = executable('qa_reader_writer_lock', 'qa_reader_writer_lock.cpp', dependencies : [ deps ], install: false)
test('qa_reader_writer_lock', qa_reader_writer_lock_exe)

qa_scheduler_exe = executable('qa_scheduler', 'qa_scheduler.cpp', dependencies : [ deps ], install: false)
test('qa_scheduler', qa_scheduler_exe)

qa_settings_exe = executable('qa_settings', 'qa_settings.cpp', dependencies : [ deps ], install: false)
test('qa_settings', qa_settings_exe)

qa_tags_exe = executable('qa_tags', 'qa_tags.cpp', dependencies : [ deps ], install: false)
test('qa_tags', qa_tags_exe)

qa_thread_affinity_exe = executable('qa_thread_affinity', 'qa_thread_affinity.cpp', dependencies : [ deps ], install: false)
test('qa_thread_affinity', qa_thread_affinity_exe)

qa_thread_pool_exe = executable('qa_thread_pool', 'qa_thread_pool.cpp', dependencies : [ deps ], install: false)
test('qa_thread_pool', qa_thread_pool_exe)

# non-UT unittests
app_deps = [gr_core_dep, fmt_dep, reflcpp_dep, gr_basic_dep, gr_testing_dep]

app_grc_exe = executable('app_grc', 'app_grc.cpp', dependencies : [ app_deps ], install: false)
test('app_grc', app_grc_exe)

app_plugins_test_exe = executable('app_plugins_test', 'app_plugins_test.cpp', dependencies : [ app_deps ], install: false)
test('app_plugins_test', app_plugins_test_exe)
