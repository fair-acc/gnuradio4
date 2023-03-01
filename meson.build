project('graph-prototype', 'cpp',
  version : '0.1',
  default_options : ['warning_level=3', 'cpp_std=gnu++20'])

fmt_dep = dependency('fmt', version:'8.1.1')
ut_dep = dependency('boost.ut')

cmake = import('cmake')
libreflcpp = cmake.subproject('refl-cpp')
reflcpp_dep = libreflcpp.dependency('refl-cpp')

subdir('bench')
