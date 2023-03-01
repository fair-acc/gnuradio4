include_directories = ['../include']

buffer_test_exe = executable('qa_buffer', 'qa_buffer.cpp',
  include_directories : include_directories, 
  dependencies : [fmt_dep, ut_dep],
  install: false)
test('qa_buffer', buffer_test_exe)

ut_test_names = [
    'qa_dynamic_port',
]
deps = [fmt_dep, ut_dep, reflcpp_dep]
cpp_args = '-fsanitize=address'
foreach test_name : ut_test_names
ut_test_exe = executable(test_name, test_name + '.cpp',
  cpp_args : cpp_args,
  link_args : cpp_args,
  include_directories : include_directories, 
  dependencies : [deps],
  install: false)
test(test_name, ut_test_exe)
endforeach
