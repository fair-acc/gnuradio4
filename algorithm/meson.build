gr_algorithm_dep = declare_dependency(include_directories: './include/', dependencies : [fftw_dep])

header_files = [
    'include/gnuradio-4.0/algorithm/fourier/fft.hpp',
    'include/gnuradio-4.0/algorithm/fourier/fft_common.hpp',
    'include/gnuradio-4.0/algorithm/fourier/fft_types.hpp',
    'include/gnuradio-4.0/algorithm/fourier/fftw.hpp',
    'include/gnuradio-4.0/algorithm/fourier/window.hpp'
]
install_headers(header_files, preserve_path: true)

if (get_option('enable_testing'))
    subdir('test')
endif
