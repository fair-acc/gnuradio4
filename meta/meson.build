gr_meta_dep = declare_dependency(include_directories: './include/', dependencies : [fmt_dep, reflcpp_dep, vir_dep])

header_files = [
    'include/gnuradio-4.0/meta/typelist.hpp',
    'include/gnuradio-4.0/meta/utils.hpp'
]
install_headers(header_files, subdir : 'graph-prototype', preserve_path: true)

if (get_option('enable_testing'))
    subdir('test')
endif
