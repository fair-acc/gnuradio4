deps = [graph_dep, fmt_dep, ut_dep]
buffer_test_exe = executable('qa_buffer', 'qa_buffer.cpp',
  dependencies : deps,
  install: false)
test('qa_buffer', buffer_test_exe)

ut_test_names = [
    'qa_dynamic_port',
]
deps = deps + reflcpp_dep

foreach test_name : ut_test_names
ut_test_exe = executable(test_name, test_name + '.cpp',
  override_options: ['b_sanitize=address'],
  dependencies : deps,
  install: false)
test(test_name, ut_test_exe)
endforeach
