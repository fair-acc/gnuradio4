
bm_names = [
    'bm_case0',
    'bm_case1'
]


deps = [graph_dep, fmt_dep, ut_dep, reflcpp_dep]
# Detect if we are using Emscripten
emscripten = meson.get_compiler('cpp').get_id() == 'emscripten'

if emscripten
  cpp_args = ''  # or set some other appropriate flag for Emscripten
else
  cpp_args = '-march=native'
endif

foreach bm_name : bm_names
bm_exe = executable(bm_name, bm_name + '.cpp',
  cpp_args : cpp_args,
  link_args : cpp_args,
  dependencies : [deps],
  install: false)
endforeach