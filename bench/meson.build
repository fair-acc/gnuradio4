bm_dep = declare_dependency(include_directories: '.', dependencies : [ut_dep])

if meson.get_compiler('cpp').get_id() == 'emscripten'
  cpp_args = ''  # or set some other appropriate flag for Emscripten
else
  cpp_args = '-march=native'
endif

bm_example = executable('bm_example', 'bm_example.cpp',
  cpp_args : cpp_args,
  link_args : cpp_args,
  dependencies : [fmt_dep, bm_dep],
  install: false)