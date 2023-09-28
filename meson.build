project('graph-prototype', 'cpp',
  version : '0.1',
  default_options : ['warning_level=0', 'cpp_std=gnu++20'])

clang_warnings = [ # '-Werror', # TODO: enable once the flags are set for the graph-prototype only and not also it's depenencies
'-Wall', '-Wextra', '-Wshadow','-Wnon-virtual-dtor','-Wold-style-cast','-Wcast-align','-Wunused','-Woverloaded-virtual','-Wpedantic','-Wconversion','-Wsign-conversion','-Wnull-dereference','-Wdouble-promotion','-Wformat=2','-Wno-unknown-pragmas','-Wimplicit-fallthrough']
gcc_warnings = clang_warnings + [ '-Wno-dangling-reference', # TODO: remove this once the fmt dangling reference bug is fixed
'-Wmisleading-indentation','-Wduplicated-cond','-Wduplicated-branches','-Wlogical-op','-Wuseless-cast','-Wno-interference-size']

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
    graph_prototype_options = ['-s','ALLOW_MEMORY_GROWTH=1','-fwasm-exceptions','-pthread','SHELL:-s PTHREAD_POOL_SIZE=30"']
  endif 
endif

# Determine compiler flags based on build type
all_compiler_flags = ''
if get_option('buildtype') == 'debug'
    all_compiler_flags = '-g'
elif get_option('buildtype') == 'release'
    all_compiler_flags = '-O3'
elif get_option('buildtype') == 'debugoptimized'
    all_compiler_flags = '-O2 -g'
elif get_option('buildtype') == 'minsize'
    all_compiler_flags = '-Os'
endif

# Additional compiler flags from the global arguments
all_compiler_flags += ' ' + ' '.join(gcc_warnings) + ' '.join(graph_prototype_options)

# Configure the header file
config_h_data = configuration_data()
config_h_data.set('CMAKE_CXX_COMPILER', compiler.get_id())
config_h_data.set('CMAKE_CXX_COMPILER_ARG1', '')
config_h_data.set('CMAKE_CXX_COMPILER_ID', compiler.get_id())
config_h_data.set('CMAKE_CXX_COMPILER_VERSION', compiler.version())
config_h_data.set('ALL_COMPILER_FLAGS', all_compiler_flags)
configure_file(input : 'cmake/config.h.in', output : 'config.h', configuration : config_h_data)

subdir('include')
subdir('src')
if (get_option('enable_testing'))
  subdir('test')
  subdir('bench')
endif
