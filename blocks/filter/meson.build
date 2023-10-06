gr_filter_dep = declare_dependency(include_directories: './include/', dependencies : [gr_core_dep])

header_files = [
    'include/gnuradio-4.0/filter/time_domain_filter.hpp'
]
install_headers(header_files, preserve_path: true)

if (get_option('enable_testing'))
    subdir('test')
endif
