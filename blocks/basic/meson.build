gr_basic_dep = declare_dependency(include_directories: './include/', dependencies : [gr_core_dep])

header_files = [
    'include/gnuradio-4.0/basic/clock_source.hpp',
    'include/gnuradio-4.0/basic/common_nodes.hpp',
    'include/gnuradio-4.0/basic/data_sink.hpp',
    'include/gnuradio-4.0/basic/function_generator.h',
    'include/gnuradio-4.0/basic/selector.hpp'
]
install_headers(header_files, preserve_path: true)

if (get_option('enable_testing'))
    subdir('test')
endif
