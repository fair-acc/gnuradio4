project('graph-prototype', 'cpp',
  version : '0.1',
  default_options : ['warning_level=0', 'cpp_std=gnu++20'])

clang_warnings = ['-Wall', '-Wextra', '-Wshadow','-Wnon-virtual-dtor','-Wold-style-cast','-Wcast-align','-Wunused','-Woverloaded-virtual','-Wpedantic','-Wconversion','-Wsign-conversion','-Wnull-dereference','-Wdouble-promotion','-Wformat=2','-Wno-unknown-pragmas','-Wimplicit-fallthrough']
gcc_warnings = clang_warnings + ['-Wmisleading-indentation','-Wduplicated-cond','-Wduplicated-branches','-Wlogical-op','-Wuseless-cast','-Wno-interference-size']

compiler = meson.get_compiler('cpp')
if compiler.get_id() == 'gcc'
    message('Compiler: GCC')
    add_global_arguments(gcc_warnings, language: 'cpp')
elif compiler.get_id() == 'clang'
    message('Compiler: LLVM/clang')
    add_global_arguments(gcc_warnings, language: 'cpp')
endif


fmt_dep = dependency('fmt', version:'8.1.1')
ut_dep = dependency('boost.ut')

cmake = import('cmake')
libreflcpp = cmake.subproject('refl-cpp')
reflcpp_dep = libreflcpp.dependency('refl-cpp')
libpmtv = subproject('pmt')
pmt_dep = libpmtv.get_variable('pmt_dep')


graph_prototype_options = []
if meson.is_cross_build()
  if meson.get_external_property('EMSCRIPTEN', false)
    graph_prototype_options = ['-s','ALLOW_MEMORY_GROWTH=1']
  endif 
endif

subdir('include')
subdir('src')
if (get_option('enable_testing'))
  subdir('test')
  subdir('bench')
endif
